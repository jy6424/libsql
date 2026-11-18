// SPDX-License-Identifier: MIT

#ifndef LIBSQL_PAGE_HEADER_H
#define LIBSQL_PAGE_HEADER_H

typedef struct sqlite3_pcache_page sqlite3_pcache_page;
typedef struct Pager Pager;
typedef struct libsql_pghdr PgHdr;
typedef struct PCache PCache;

/*
** Every page in the cache is controlled by an instance of the following
** structure.
*/
struct libsql_pghdr {
  sqlite3_pcache_page *pPage;    /* Pcache object page handle */
  void *pData;                   /* Page data */
  void *pExtra;                  /* Extra content */
  PCache *pCache;                /* PRIVATE: Cache that owns this page */
  PgHdr *pDirty;                 /* Transient list of dirty sorted by pgno */
  Pager *pPager;                 /* The pager this page is part of */
  unsigned int pgno;             /* Page number for this page */
  unsigned int pageHash;         /* Hash of page content */
  unsigned short flags;          /* PGHDR flags defined below */
  /* What changed in this page, for physiological logging - see the PagerDelta
  ** comment in pager.c.  Held inline rather than behind a pointer because a
  ** page outlives the transaction that wrote it: anything allocated per
  ** transaction and freed at its end would leave this header pointing at
  ** released memory, and plogGen could not then be read to find that out.
  ** plogNRange is meaningless unless plogGen is the pager's current
  ** generation; plogWhole and an empty range list both mean the whole page. */
  unsigned int plogGen;          /* Transaction these ranges describe */
  unsigned char plogWhole;       /* Log the whole page, ignore the ranges */
  unsigned char plogNRange;      /* Pairs in plogRange */
  /* Set by the pager when the last 8 bytes of this page are reserved bytes
  ** that the log may stamp with a page LSN, and are already declared as a
  ** range.  The pager owns the knowledge of how many bytes are reserved; the
  ** log only needs to be told that the slot is there.  See plogFrames(). */
  unsigned char plogLsn;         /* Stamp a page LSN into the reserved bytes */
  unsigned short plogRange[2*8]; /* {offset, length} pairs */

  /**********************************************************************
  ** Elements above, except pCache, are public.  All that follow are 
  ** private to pcache.c and should not be accessed by other modules.
  ** pCache is grouped with the public elements for efficiency.
  */
  unsigned long long nRef;       /* Number of users of this page */
  PgHdr *pDirtyNext;             /* Next element in list of dirty pages */
  PgHdr *pDirtyPrev;             /* Previous element in list of dirty pages */
                          /* NB: pDirtyNext and pDirtyPrev are undefined if the
                          ** PgHdr object is not dirty */
};

typedef struct libsql_pghdr libsql_pghdr;

#endif // LIBSQL_PAGE_HEADER_H
