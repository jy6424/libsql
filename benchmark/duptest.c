/*
** duptest -- measure how much of an LM-DiskANN index is duplicated
** neighbour vectors, and project what splitting them out would save.
**
** In the LM-DiskANN layout every node block carries a copy of the
** (compressed) vector of each of its neighbours:
**
**   [u64 rowid][u16 nEdges][6 pad][node vector]
**   [edge vector] * nMaxEdges
**   ([u32 unused][f32 distance][u64 edgeId]) * nMaxEdges
**
** So a vector is stored once as the node's own copy and once more in the
** block of every node that lists it as a neighbour -- in-degree + 1 copies
** in all.  DiskANN instead keeps one compressed vector per node in a
** separate list and leaves only neighbour ids in the block.  This tool
** scans an existing index and reports what that change would be worth,
** before anything is modified.
**
** It only reads, so it can run against a database a benchmark just built.
**
**   cc -o duptest duptest.c -lsqlite3        (or link libsql)
**   ./duptest sift.db [index_name]
**
** The layout constants below mirror vectorIndexInt.h, vectordiskann.c and
** vector.c.  They are checked against the real block size read from the
** database, so a drift shows up as a reported mismatch rather than as
** quietly wrong numbers.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sqlite3.h"

/* Parameter tags - vectorIndexInt.h */
#define P_FORMAT    1
#define P_TYPE      3
#define P_DIM       4
#define P_BLOCK     6
#define P_MAXNB    10
#define P_COMPRESS 11

/* Vector types - vectorInt.h */
#define T_F32   1
#define T_F64   2
#define T_1BIT  3
#define T_F8    4
#define T_F16   5
#define T_B16   6

#define EDGE_META_SIZE 16        /* [u32 unused][f32 distance][u64 edgeId] */

/* From v4 the neighbour vectors are in <idx>_cvecshadow, not in the block. */
#define IS_SPLIT(fmt) ((fmt) >= 4)

typedef sqlite3_int64 i64;
typedef unsigned char u8;

static const char *typeName(int t){
  switch( t ){
    case T_F32:  return "float32";
    case T_F64:  return "float64";
    case T_1BIT: return "float1bit";
    case T_F8:   return "float8";
    case T_F16:  return "float16";
    case T_B16:  return "floatb16";
  }
  return "?";
}

/* vector.c:vectorDataSize() */
static int vecDataSize(int type, int dims){
  switch( type ){
    case T_F32:  return dims*4;
    case T_F64:  return dims*8;
    case T_1BIT: return (dims+7)/8;
    case T_F8:   return ((dims+3)/4)*4 + 4 + 4;   /* ALIGN(dims,4)+alpha+shift */
    case T_F16:  return dims*2;
    case T_B16:  return dims*2;
  }
  return 0;
}

/* vectordiskann.c:nodeMetadataSize() */
static int nodeMetaSize(int fmt){ return fmt<=2 ? 8+2 : 8+8; }

/*
** Parameters are a run of 9-byte records: one tag byte then a little-endian
** u64.  The last record for a tag wins, as in vectorIdxParamsGetU64().
*/
static sqlite3_uint64 paramGet(const u8 *a, int n, int tag){
  sqlite3_uint64 v = 0;
  int i, j;
  for(i=0; i+9<=n; i+=9){
    if( a[i]!=tag ) continue;
    v = 0;
    for(j=0; j<8; j++) v |= ((sqlite3_uint64)a[i+1+j]) << (8*j);
  }
  return v;
}

static sqlite3_uint64 rdLE16(const u8 *a){
  return (sqlite3_uint64)a[0] | ((sqlite3_uint64)a[1]<<8);
}
static sqlite3_uint64 rdLE64(const u8 *a){
  sqlite3_uint64 v = 0; int j;
  for(j=0; j<8; j++) v |= ((sqlite3_uint64)a[j]) << (8*j);
  return v;
}

/*
** Per-node counters, indexed by shadow-table rowid.
**   aIn[r]  - how many other blocks hold a copy of r's vector (in-degree),
**             which is exactly how many times r is duplicated
**   aOut[r] - how many edges r's own block holds
*/
static unsigned int *aIn = 0;
static unsigned int *aOut = 0;
static i64 nTrackAlloc = 0, mxRowid = 0, nOutOfRange = 0;

/* Optional per-node CSV, opened by main() when -o is given. */
static FILE *csvOut = 0;

/* Make room for rowid r in both arrays.  Returns 0 if r cannot be tracked. */
static int nodeSlot(i64 r){
  if( r<=0 || r>100000000 ) return 0;
  if( r>=nTrackAlloc ){
    i64 nNew = nTrackAlloc ? nTrackAlloc : 65536;
    unsigned int *a1, *a2;
    while( nNew<=r ) nNew *= 2;
    a1 = realloc(aIn, (size_t)nNew*sizeof(unsigned int));
    if( a1==0 ) return 0;
    aIn = a1;
    a2 = realloc(aOut, (size_t)nNew*sizeof(unsigned int));
    if( a2==0 ) return 0;        /* aIn is simply larger than we admit to */
    aOut = a2;
    memset(aIn  + nTrackAlloc, 0, (size_t)(nNew-nTrackAlloc)*sizeof(unsigned int));
    memset(aOut + nTrackAlloc, 0, (size_t)(nNew-nTrackAlloc)*sizeof(unsigned int));
    nTrackAlloc = nNew;
  }
  if( r>mxRowid ) mxRowid = r;
  return 1;
}

#define MB(x) ((double)(x)/1048576.0)

static int analyse(sqlite3 *db, const char *zIdx){
  char zSql[512];
  sqlite3_stmt *pStmt = 0;
  const u8 *aParam;
  int nParam, rc;
  int fmt, type, dims, nbType, nodeVec, edgeVec, nodeMeta;
  int nMaxEdge, metaOff, blockParam, blockReal = 0;
  i64 nNode = 0, nEdge = 0, i;
  i64 nCVecRow = 0, nCVecByte = 0;
  i64 nDegHist[65];
  int mxDeg = 0;

  memset(nDegHist, 0, sizeof(nDegHist));

  /* Counters are per index, so clear whatever a previous one left. */
  if( nTrackAlloc ){
    memset(aIn,  0, (size_t)nTrackAlloc*sizeof(unsigned int));
    memset(aOut, 0, (size_t)nTrackAlloc*sizeof(unsigned int));
  }
  mxRowid = 0;
  nOutOfRange = 0;

  sqlite3_snprintf(sizeof(zSql), zSql,
    "SELECT metadata FROM libsql_vector_meta_shadow WHERE name = %Q", zIdx);
  rc = sqlite3_prepare_v2(db, zSql, -1, &pStmt, 0);
  if( rc!=SQLITE_OK || sqlite3_step(pStmt)!=SQLITE_ROW ){
    fprintf(stderr, "no index named '%s' in libsql_vector_meta_shadow\n", zIdx);
    sqlite3_finalize(pStmt);
    return 1;
  }
  aParam = sqlite3_column_blob(pStmt, 0);
  nParam = sqlite3_column_bytes(pStmt, 0);

  fmt    = (int)paramGet(aParam, nParam, P_FORMAT);
  type   = (int)paramGet(aParam, nParam, P_TYPE);
  dims   = (int)paramGet(aParam, nParam, P_DIM);
  nbType = (int)paramGet(aParam, nParam, P_COMPRESS);
  blockParam = (int)paramGet(aParam, nParam, P_BLOCK);
  if( fmt==0 ) fmt = 1;
  if( nbType==0 ) nbType = type;    /* unset means "same as the column" */
  sqlite3_finalize(pStmt);

  nodeVec  = vecDataSize(type, dims);
  edgeVec  = vecDataSize(nbType, dims);
  nodeMeta = nodeMetaSize(fmt);

  /* The stored block-size parameter has had a couple of encodings, so take
  ** the real size from a row and use the parameter only as a cross-check. */
  sqlite3_snprintf(sizeof(zSql), zSql,
                   "SELECT length(data) FROM \"%w_shadow\" LIMIT 1", zIdx);
  if( sqlite3_prepare_v2(db, zSql, -1, &pStmt, 0)==SQLITE_OK
   && sqlite3_step(pStmt)==SQLITE_ROW ){
    blockReal = sqlite3_column_int(pStmt, 0);
  }
  sqlite3_finalize(pStmt);
  if( blockReal<=0 ){
    fprintf(stderr, "shadow table '%s_shadow' is empty or missing\n", zIdx);
    return 1;
  }

  if( IS_SPLIT(fmt) ){
    nMaxEdge = (blockReal - (nodeVec + nodeMeta)) / EDGE_META_SIZE;
    metaOff  = nodeMeta + nodeVec;
  }else{
    nMaxEdge = (blockReal - (nodeVec + nodeMeta)) / (edgeVec + EDGE_META_SIZE);
    metaOff  = nodeMeta + nodeVec + nMaxEdge*edgeVec;
  }

  printf("index '%s'\n", zIdx);
  printf("  format v%d  dims %d  node %s (%d B)  neighbour %s (%d B)\n",
         fmt, dims, typeName(type), nodeVec, typeName(nbType), edgeVec);
  printf("  block %d B", blockReal);
  if( blockParam && blockParam!=blockReal ){
    printf("  (parameter says %d - encodings differ, using the real one)",
           blockParam);
  }
  printf("\n");
  printf("  slots %d   layout: hdr %d | node vec %d | edge vecs %d | edge meta %d\n",
         nMaxEdge, nodeMeta, nodeVec,
         IS_SPLIT(fmt) ? 0 : nMaxEdge*edgeVec,
         nMaxEdge*EDGE_META_SIZE);
  if( IS_SPLIT(fmt) ){
    printf("  neighbour vectors are split out into %s_cvecshadow\n", zIdx);
    sqlite3_snprintf(sizeof(zSql), zSql,
      "SELECT count(*), total(length(data)) FROM \"%w_cvecshadow\"", zIdx);
    if( sqlite3_prepare_v2(db, zSql, -1, &pStmt, 0)==SQLITE_OK
     && sqlite3_step(pStmt)==SQLITE_ROW ){
      nCVecRow  = sqlite3_column_int64(pStmt, 0);
      nCVecByte = (i64)sqlite3_column_double(pStmt, 1);
    }
    sqlite3_finalize(pStmt);
  }

  if( nMaxEdge<=0 || metaOff + nMaxEdge*EDGE_META_SIZE > blockReal ){
    fprintf(stderr, "  !! layout does not fit the block - constants have drifted\n");
    return 1;
  }

  /* Scan every node: count live edges and tally in-degrees. */
  sqlite3_snprintf(sizeof(zSql), zSql,
                   "SELECT rowid, data FROM \"%w_shadow\"", zIdx);
  rc = sqlite3_prepare_v2(db, zSql, -1, &pStmt, 0);
  if( rc!=SQLITE_OK ){
    fprintf(stderr, "cannot scan shadow table: %s\n", sqlite3_errmsg(db));
    return 1;
  }
  while( sqlite3_step(pStmt)==SQLITE_ROW ){
    i64 self = sqlite3_column_int64(pStmt, 0);
    const u8 *aBlk = sqlite3_column_blob(pStmt, 1);
    int nBlk = sqlite3_column_bytes(pStmt, 1);
    int nEdges, j;
    if( aBlk==0 || nBlk<blockReal ) continue;
    nNode++;
    nEdges = (int)rdLE16(aBlk + 8);
    if( nEdges>nMaxEdge ) nEdges = nMaxEdge;   /* corrupt or format skew */
    nEdge += nEdges;
    if( nEdges>mxDeg ) mxDeg = nEdges;
    nDegHist[nEdges<64 ? nEdges : 64]++;
    if( nodeSlot(self) ) aOut[self] = (unsigned int)nEdges;
    for(j=0; j<nEdges; j++){
      /* Each edge is one copy of that neighbour's vector living here. */
      i64 r = (i64)rdLE64(aBlk + metaOff + j*EDGE_META_SIZE + 8);
      if( nodeSlot(r) ) aIn[r]++; else nOutOfRange++;
    }
  }
  sqlite3_finalize(pStmt);

  if( nNode==0 ){
    fprintf(stderr, "  no rows scanned\n");
    return 1;
  }

  /* ---- duplication ---- */
  {
    i64 nAlloc = nNode*(i64)nMaxEdge;      /* copies the file has room for */
    i64 nLive  = nEdge;                    /* copies actually in use */
    i64 nZero = 0, nMaxIn = 0;
    i64 aTopId[10]; unsigned int aTopN[10];
    i64 aInHist[65];
    double meanIn;
    int t;

    memset(aInHist, 0, sizeof(aInHist));
    memset(aTopN, 0, sizeof(aTopN));
    for(t=0; t<10; t++) aTopId[t] = 0;

    for(i=1; i<=mxRowid; i++){
      unsigned int d;
      if( aIn==0 ) break;
      d = aIn[i];
      if( d==0 && aOut && aOut[i]==0 ) continue;   /* rowid never existed */
      if( d==0 ) nZero++;
      if( d>nMaxIn ) nMaxIn = d;
      aInHist[d<64 ? d : 64]++;
      /* keep the ten most-copied nodes, insertion sort into a tiny table */
      if( d>aTopN[9] ){
        for(t=9; t>0 && aTopN[t-1]<d; t--){
          aTopN[t] = aTopN[t-1]; aTopId[t] = aTopId[t-1];
        }
        aTopN[t] = d; aTopId[t] = i;
      }
    }
    meanIn = (double)nLive/(double)nNode;

    printf("\nscan: %lld nodes, %lld live edges  (mean out-degree %.2f, max %d)\n",
           (long long)nNode, (long long)nEdge, meanIn, mxDeg);
    printf("  out-degree: full slots %lld (%.1f%%), empty %lld\n",
           (long long)nDegHist[nMaxEdge<64 ? nMaxEdge : 64],
           100.0*nDegHist[nMaxEdge<64 ? nMaxEdge : 64]/(double)nNode,
           (long long)nDegHist[0]);
    printf("  in-degree : mean %.2f, max %lld, never referenced %lld",
           meanIn, (long long)nMaxIn, (long long)nZero);
    if( nOutOfRange ) printf(", out of range %lld", (long long)nOutOfRange);
    printf("\n");

    /* How lopsided the copying is decides whether a per-node scheme can
    ** pay off, so show the shape and not just the average. */
    printf("  in-degree histogram (copies per node):");
    for(i=0; i<=64; i++){
      if( aInHist[i] ){
        printf(" %lld%s:%lld", (long long)i, i==64 ? "+" : "",
               (long long)aInHist[i]);
      }
    }
    printf("\n  most-copied nodes:");
    for(t=0; t<10 && aTopN[t]; t++){
      printf(" %lld(%u)", (long long)aTopId[t], aTopN[t]);
    }
    printf("\n");

    printf("\nneighbour-vector copies (%d B each)\n", edgeVec);
    if( IS_SPLIT(fmt) ){
      printf("  stored in side table : %lld  (%.1f MB)   one per node\n",
             (long long)nCVecRow, MB(nCVecByte));
      printf("  v1-v3 would have had : %lld  (%.1f MB)\n",
             (long long)nLive, MB(nLive*(i64)edgeVec));
      printf("  duplication          : 1.0x  (was %.1fx)\n",
             (double)nLive/(double)nNode);
      if( nCVecRow!=nNode ){
        printf("  !! %lld nodes but %lld side-table rows - they should match\n",
               (long long)nNode, (long long)nCVecRow);
      }
    }else{
      printf("  slots allocated : %lld  (%.1f MB)\n",
             (long long)nAlloc, MB(nAlloc*(i64)edgeVec));
      printf("  actually in use : %lld  (%.1f MB)\n",
             (long long)nLive, MB(nLive*(i64)edgeVec));
      printf("  needed if split : %lld  (%.1f MB)   one per node\n",
             (long long)nNode, MB(nNode*(i64)edgeVec));
      printf("  duplication     : %.1fx allocated, %.1fx live\n",
             (double)nAlloc/(double)nNode, (double)nLive/(double)nNode);
    }
  }

  /* ---- projection ---- */
  {
    i64 cur = nNode*(i64)blockReal;
    i64 nodeRegion = nNode*(i64)(nodeMeta + nodeVec);
    i64 copyRegion = nNode*(i64)nMaxEdge*edgeVec;
    i64 metaRegion = nNode*(i64)nMaxEdge*EDGE_META_SIZE;
    int id8 = 8, id4 = 4;
    i64 new8 = nNode*(i64)(nodeMeta + nodeVec + nMaxEdge*id8) + nNode*(i64)edgeVec;
    i64 new4 = nNode*(i64)(nodeMeta + nodeVec + nMaxEdge*id4) + nNode*(i64)edgeVec;

    if( IS_SPLIT(fmt) ){
      /* Already split: show what it cost and what it saved. */
      i64 side = nCVecByte ? nCVecByte : nNode*(i64)edgeVec;
      i64 tot = cur + side;
      i64 asV3 = nNode*(i64)(nodeMeta + nodeVec
                             + nMaxEdge*(edgeVec + EDGE_META_SIZE));
      printf("\nindex size now: %.1f MB\n", MB(tot));
      printf("  node blocks              : %7.1f MB  (%.1f%%)\n",
             MB(cur), 100.0*cur/(double)tot);
      printf("    node header + own vec  : %7.1f MB\n", MB(nodeRegion));
      printf("    edge metadata          : %7.1f MB\n", MB(metaRegion));
      printf("  neighbour vector table   : %7.1f MB  (%.1f%%)   one per node\n",
             MB(side), 100.0*side/(double)tot);
      printf("\nsame graph in the v1-v3 layout would be: %.1f MB   (%.2fx larger)\n",
             MB(asV3), (double)asV3/(double)tot);
      printf("\n  the node's own full vector is now %.0f%% of the index, and it\n"
             "  is also in the base table - dropping it is the next lever.\n",
             100.0*nodeRegion/(double)tot);
      goto detail;
    }

    printf("\nindex size now: %.1f MB\n", MB(cur));
    printf("  node header + own vector : %7.1f MB  (%.1f%%)\n",
           MB(nodeRegion), 100.0*nodeRegion/(double)cur);
    printf("  neighbour vector copies  : %7.1f MB  (%.1f%%)   <- duplicated\n",
           MB(copyRegion), 100.0*copyRegion/(double)cur);
    printf("  edge metadata            : %7.1f MB  (%.1f%%)\n",
           MB(metaRegion), 100.0*metaRegion/(double)cur);

    printf("\nif neighbour vectors are split out (DiskANN layout):\n");
    printf("  node block = hdr + own vector + %d ids, plus one shared\n"
           "  compressed vector per node\n", nMaxEdge);
    printf("  with u64 ids : %7.1f MB   %.2fx smaller\n",
           MB(new8), (double)cur/(double)new8);
    printf("  with u32 ids : %7.1f MB   %.2fx smaller\n",
           MB(new4), (double)cur/(double)new4);
    printf("\n  note: this counts the index only.  The full vectors still\n"
           "  live in the base table either way, so the comparison is\n"
           "  like for like.\n");
  }

detail:
  /* Per-node detail, for looking at the distribution properly. */
  if( csvOut && aIn ){
    for(i=1; i<=mxRowid; i++){
      if( aIn[i]==0 && (aOut==0 || aOut[i]==0) ) continue;
      fprintf(csvOut, "%s,%lld,%u,%u,%lld\n", zIdx, (long long)i,
              aIn[i], aOut ? aOut[i] : 0,
              (long long)aIn[i]*edgeVec);
    }
    fflush(csvOut);
  }
  return 0;
}

int main(int argc, char **argv){
  sqlite3 *db = 0;
  int rc, nFail = 0, iArg = 1;
  const char *zCsv = 0;

  while( iArg<argc && argv[iArg][0]=='-' ){
    if( strcmp(argv[iArg], "-o")==0 && iArg+1<argc ){
      zCsv = argv[++iArg];
      iArg++;
    }else{
      fprintf(stderr, "unknown option: %s\n", argv[iArg]);
      return 2;
    }
  }
  if( argc-iArg<1 ){
    fprintf(stderr,
      "usage: %s [-o nodes.csv] <database> [index_name ...]\n"
      "  -o  write one row per node: index,rowid,in_degree,out_degree,copy_bytes\n",
      argv[0]);
    return 2;
  }
  if( zCsv ){
    csvOut = fopen(zCsv, "w");
    if( csvOut==0 ){
      fprintf(stderr, "cannot write %s\n", zCsv);
      return 2;
    }
    fprintf(csvOut, "index,rowid,in_degree,out_degree,copy_bytes\n");
  }
  argv += iArg - 1;          /* keep argv[1] as the database path */
  argc -= iArg - 1;
  rc = sqlite3_open_v2(argv[1], &db, SQLITE_OPEN_READONLY, 0);
  if( rc!=SQLITE_OK ){
    fprintf(stderr, "cannot open %s: %s\n", argv[1], sqlite3_errmsg(db));
    return 2;
  }

  if( argc>2 ){
    int i;
    for(i=2; i<argc; i++){
      if( i>2 ) printf("\n");
      nFail += analyse(db, argv[i]);
    }
  }else{
    /* No index named: do every one the database knows about. */
    sqlite3_stmt *pStmt = 0;
    int n = 0;
    rc = sqlite3_prepare_v2(db,
           "SELECT name FROM libsql_vector_meta_shadow ORDER BY name", -1,
           &pStmt, 0);
    if( rc!=SQLITE_OK ){
      fprintf(stderr, "no libsql_vector_meta_shadow in %s\n", argv[1]);
      sqlite3_close(db);
      return 2;
    }
    while( sqlite3_step(pStmt)==SQLITE_ROW ){
      const char *z = (const char*)sqlite3_column_text(pStmt, 0);
      char zName[256];
      sqlite3_snprintf(sizeof(zName), zName, "%s", z ? z : "");
      if( n++ ) printf("\n");
      nFail += analyse(db, zName);
    }
    sqlite3_finalize(pStmt);
    if( n==0 ) fprintf(stderr, "no vector indexes in %s\n", argv[1]);
  }

  sqlite3_close(db);
  if( csvOut ) fclose(csvOut);
  free(aIn);
  free(aOut);
  return nFail ? 1 : 0;
}
