#ifndef _MAIN_H
#define _MAIN_H 
#define BLOCK_SIZE 2
# if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && !defined(EXTRALARGE_DATASET) && !defined(NOTEXTRA_BUT_LARGE_DATASET)
// #define LARGE_DATASET
#define LARGE_DATASET
# endif
# if !defined(NI) && !defined(NJ) && !defined(NK) && !defined(NL)
# ifdef MINI_DATASET
#define NI 16
#define NJ 18
#define NK 22
#define NL 24
# endif
# ifdef SMALL_DATASET
#define NI 40
#define NJ 50
#define NK 70
#define NL 80
# endif
# ifdef MEDIUM_DATASET
#define NI 180
#define NJ 190
#define NK 210
#define NL 220
# endif
# ifdef LARGE_DATASET
// #define NI 800
// #define NJ 900
// #define NK 1100
// #define NL 1200
#define NI 1600
#define NJ 1600
#define NK 1600
#define NL 1600
# endif
# ifdef EXTRALARGE_DATASET
// #define NI 1600
// #define NJ 1800
// #define NK 2200
// #define NL 2400
#define NI 5000
#define NJ 5000
#define NK 5000
#define NL 5000
# endif
#endif
#endif