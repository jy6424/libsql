#!/bin/sh
#
# Page LSN test, run against your own SQL file with the ordinary benchtest.
#
#   ./lsn_test.sh [sqlfile] [kill_after_seconds]
#
# Default sqlfile is dataset/insert_sift100k.sql.
#
# Three arms:
#   A  no LSN      - baseline size and insert time
#   B  LSN on      - same data, plus the stamp; prints the time it costs
#   C  crash+wreck - kill a run so the log keeps delta frames, overwrite the
#                    stamps in the database file, then read.  The read has to
#                    fail: that is the whole point of the stamp.
#
set -u
cd "$(dirname "$0")"

SQL=${1:-dataset/insert_sift100k.sql}
KILL_AFTER=${2:-25}
LIB=../.libs
FAIL=0

[ -f "$SQL" ] || { echo "no such sql file: $SQL"; exit 2; }

# benchtest is cheap, so always rebuild it rather than trust its mtime.
cc benchtest.c -o benchtest -L$LIB -llibsql -O2 \
  || { echo "failed to build benchtest"; exit 2; }

# The library is not cheap, so just check it is not older than the sources
# that matter.  A stale .so is the one failure mode that makes every check
# below fail for a reason that has nothing to do with the code under test.
for src in ../src/plog.c ../src/pager.c ../src/page_header.h; do
  for so in $LIB/*.so $LIB/*.dylib; do
    [ -e "$so" ] || continue
    if [ "$src" -nt "$so" ]; then
      echo "stale library: $src is newer than $so"
      echo "rebuild it first:   (cd .. && make -j)"
      exit 2
    fi
  done
done

# sqlite3 shell, for the read in arm C.  Optional.
SHELL_BIN=""
for c in ../sqlite3 ./sqlite3 ../shell; do
  [ -x "$c" ] && SHELL_BIN="$c" && break
done

say()  { printf '%s\n' "$*"; }
rule() { printf '=== %s %s\n' "$1" "$(printf '%.0s=' $(seq 1 $((66 - ${#1}))))"; }
ok()   { if [ "$2" = 1 ]; then printf '  [PASS] %-38s %s\n' "$1" "${3:-}";
         else printf '  [FAIL] %-38s %s\n' "$1" "${3:-}"; FAIL=$((FAIL+1)); fi }

# Byte 20 of the database header is the reserved-byte count.
reserve_of() { dd if="$1" bs=1 skip=20 count=1 2>/dev/null | od -An -tu1 | tr -d ' \n'; }

# The LSN slot is the last 8 bytes of a page.  Page 2 starts at 4096.
stamp_of() { dd if="$1" bs=1 skip=$(( $2 * 4096 - 8 )) count=8 2>/dev/null \
             | od -An -tx1 | tr -d ' \n'; }

# Pull "size  : 643.6250 MB" and "insert: 5010.80 micros" out of the log.
# Anchored, or ".*insert" also matches the one inside insert_sift100k.sql and
# the number that comes back is the 100 out of the file name.
field() { sed -n "s/^ *$2 *: *\([0-9.][0-9.]*\).*/\1/p" "$1" | head -1; }

secs() { printf '%.1f' "$1"; }

run_arm() {                        # run_arm <name> <db> <env...>
  name=$1; db=$2; shift 2
  rm -f "$db" "$db"-wal "$db"-shm
  sync
  t0=$(date +%s.%N)
  env "$@" LD_LIBRARY_PATH=$LIB ./benchtest "$SQL" "$db" > "$name.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  WALL=$(echo "$t1 - $t0" | bc)
  [ $rc -eq 0 ] || { say "  benchtest failed (rc=$rc), tail of $name.log:";
                     tail -5 "$name.log"; }
  return $rc
}

say "page LSN test"
say "  sql      : $SQL"
say "  shell    : ${SHELL_BIN:-<none found, arm C read will be skipped>}"
say ""

# Fail fast if the binary predates the page-LSN work, otherwise every check
# below fails for the same uninteresting reason.
rm -f probe.db probe.db-wal probe.db-shm
printf -- '---probe\nCREATE TABLE p(x);\n' > probe.sql
LIBSQL_PLOG=1 LIBSQL_PLOG_LSN=1 LD_LIBRARY_PATH=$LIB \
  ./benchtest probe.sql probe.db > probe.log 2>&1
if ! grep -q "page LSN on" probe.log; then
  say "benchtest does not know about LIBSQL_PLOG_LSN."
  say ""
  say "The library predates the page-LSN change.  Rebuild it:"
  say "    git pull && (cd .. && make -j)"
  say ""
  say "Nothing below would mean anything until then - every check would fail"
  say "for that one reason."
  rm -f probe.db probe.db-wal probe.db-shm probe.sql
  exit 2
fi
rm -f probe.db probe.db-wal probe.db-shm probe.sql
say "  binary   : has page-LSN support"
say ""

# ---------------------------------------------------------------- arm A ----
rule "A: plog, no LSN"
run_arm plain plain.db LIBSQL_PLOG=1
A_SIZE=$(field plain.log "size")
A_INS=$(field plain.log "insert")
A_RES=$(reserve_of plain.db)
say "  size $A_SIZE MB   insert $A_INS us   wall $(secs "$WALL")s   reserve=$A_RES"
A_WALL=$WALL
ok "no reserved bytes without LSN" "$([ "$A_RES" = 0 ] && echo 1 || echo 0)" "byte20=$A_RES"
ok "wal/shm removed on close" \
   "$([ ! -e plain.db-wal ] && [ ! -e plain.db-shm ] && echo 1 || echo 0)" \
   "$(ls plain.db-wal plain.db-shm 2>/dev/null | tr '\n' ' ')"
say ""

# ---------------------------------------------------------------- arm B ----
rule "B: plog + LSN"
run_arm stamped stamped.db LIBSQL_PLOG=1 LIBSQL_PLOG_LSN=1
B_SIZE=$(field stamped.log "size")
B_INS=$(field stamped.log "insert")
B_RES=$(reserve_of stamped.db)
B_STAMP=$(stamp_of stamped.db 2)
A_STAMP=$(stamp_of plain.db 2)
say "  size $B_SIZE MB   insert $B_INS us   wall $(secs "$WALL")s   reserve=$B_RES"
say "  page 2 tail: $B_STAMP   (no-LSN arm: $A_STAMP)"
say ""
say "  cost of the stamp:"
echo "    insert  $A_INS -> $B_INS us" \
     "($(echo "scale=2; 100*($B_INS-$A_INS)/$A_INS" | bc)%)"
echo "    wall    $(secs "$A_WALL") -> $(secs "$WALL") s" \
     "($(echo "scale=2; 100*($WALL-$A_WALL)/$A_WALL" | bc)%)"
echo "    size    $A_SIZE -> $B_SIZE MB" \
     "($(echo "scale=2; 100*($B_SIZE-$A_SIZE)/$A_SIZE" | bc)%)"
say ""
ok "reserved bytes claimed"  "$([ "$B_RES" = 8 ] && echo 1 || echo 0)" "byte20=$B_RES"
ok "stamp reached the db file" \
   "$([ -n "$B_STAMP" ] && [ "$B_STAMP" != "0000000000000000" ] && echo 1 || echo 0)" "$B_STAMP"
# With no reserved bytes the same offset holds ordinary btree content, so the
# only thing worth asserting is that the two arms do not agree there.
ok "that tail is the stamp, not page content" \
   "$([ "$A_STAMP" != "$B_STAMP" ] && echo 1 || echo 0)" \
   "A=$A_STAMP B=$B_STAMP"
ok "same data as the no-LSN arm (size within 1%)" \
   "$(echo "$A_SIZE $B_SIZE" | awk '{d=($2-$1)/$1; if(d<0)d=-d; print (d<0.01)?1:0}')" \
   "$A_SIZE vs $B_SIZE MB"
ok "wal/shm removed on close" \
   "$([ ! -e stamped.db-wal ] && [ ! -e stamped.db-shm ] && echo 1 || echo 0)" \
   "$(ls stamped.db-wal stamped.db-shm 2>/dev/null | tr '\n' ' ')"
say ""

# ---------------------------------------------------------------- arm C ----
rule "C: crash, wreck the stamps, read"
rm -f wreck.db wreck.db-wal wreck.db-shm
LIBSQL_PLOG=1 LIBSQL_PLOG_LSN=1 LD_LIBRARY_PATH=$LIB \
  ./benchtest "$SQL" wreck.db > wreck.log 2>&1 &
PID=$!
say "  running for ${KILL_AFTER}s then SIGKILL (leaves the log in place)..."
sleep "$KILL_AFTER"
kill -9 $PID 2>/dev/null
wait $PID 2>/dev/null

WALSZ=$(stat -c %s wreck.db-wal 2>/dev/null || stat -f %z wreck.db-wal 2>/dev/null || echo 0)
ok "log left behind by the crash" "$([ "$WALSZ" -gt 0 ] && echo 1 || echo 0)" "${WALSZ} B"

# Overwrite every page's LSN slot.  Pages with no pending delta frame come
# straight out of the file and do not care; one with a chain must be refused.
N=$(python3 - wreck.db <<'EOF'
import sys
f=open(sys.argv[1],'r+b'); f.seek(0,2); sz=f.tell(); p=2; n=0
while p*4096 <= sz:
    f.seek(p*4096-8); f.write(b'\xa5'*8); p+=1; n+=1
f.close(); print(n)
EOF
)
ok "stamps overwritten" "$([ "${N:-0}" -gt 0 ] && echo 1 || echo 0)" "${N:-0} pages"

if [ -n "$SHELL_BIN" ]; then
  # The log was written by plog, whose header version stock wal.c rejects with
  # SQLITE_CANTOPEN.  Read it with the same implementation that wrote it, or
  # the open fails before anything gets as far as checking a stamp.
  t0=$(date +%s.%N)
  OUT=$(LIBSQL_PLOG=1 LIBSQL_PLOG_LSN=1 LD_LIBRARY_PATH=$LIB "$SHELL_BIN" wreck.db \
        "SELECT count(*) FROM sqlite_master;
         SELECT count(*) FROM sqlite_master WHERE name LIKE '%shadow%';" 2>&1)
  t1=$(date +%s.%N)
  say "  read took $(secs "$(echo "$t1 - $t0" | bc)")s"
  say "  output: $(echo "$OUT" | tr '\n' ' ' | cut -c1-110)"
  ok "read refuses the moved base" \
     "$(echo "$OUT" | grep -qi 'malformed\|corrupt' && echo 1 || echo 0)" \
     "expected 'database disk image is malformed'"
else
  say "  no sqlite3 shell found - build one and re-run to check the refusal:"
  say "    LIBSQL_PLOG=1 LD_LIBRARY_PATH=$LIB ../sqlite3 wreck.db \\"
  say "      'SELECT count(*) FROM sqlite_master;'"
fi
say ""

rule "result"
if [ "$FAIL" = 0 ]; then say "ALL PASS"; else say "FAILED ($FAIL)"; fi
say "logs: plain.log stamped.log wreck.log"
exit "$FAIL"
