#include "main.h"

static void init_array_task(int ni, int nj, int nk, int nl,
                           double *alpha,
                           double *beta,
                           double A[ni][nk],
                           double B[nj][nk],
                           double C[nl][nj],
                           double D[ni][nl]) {
  *alpha = 1.5;
  *beta = 1.2;
  double div_ni = 1.0 / ni, div_nj = 1.0 / nj, div_nl = 1.0 / nl, div_nk = 1.0 / nk;
  #pragma omp parallel shared(ni, nj, nl, nk, A, B, C, D, div_ni, div_nl, div_nj, div_nk) 
       {
              #pragma omp single
              {
               #pragma omp taskloop
                for (int i = 0; i < ni; ++i)
                  for (int j = 0; j < nk; ++j)
                    A[i][j] = (double)((i*j+1) % ni) * div_ni;

               #pragma omp taskloop
                for (int i = 0; i < nj; ++i)
                  for (int j = 0; j < nk; ++j)
                    B[i][j] = (double)(j*(i+1) % nj) * div_nj;

               #pragma omp taskloop
                for (int i = 0; i < nl; ++i)
                  for (int j = 0; j < nj; ++j)
                    C[i][j] = (double)((j*(i+3)+1) % nl) * div_nl;

               #pragma omp taskloop
                for (int i = 0; i < ni; ++i)
                  for (int j = 0; j < nl; ++j)
                    D[i][j] = (double)(i*(j+2) % nk) * div_nk;
              }
       }
}

static void kernel_2mm_task(int ni, int nj, int nk, int nl,
                            double alpha,
                            double beta,
                            double tmp[ni][nj],
                            double A[ni][nk],
                            double B[nj][nk],
                            double C[nl][nj],
                            double D[ni][nl]) {

double sum = 0.0;

#pragma omp parallel shared(ni, nj, nl, nk, tmp, A, B, C, D, alpha, beta, sum) 
       {
       #pragma omp single
              {
                     #pragma omp taskloop
                     for (int i = 0; i < ni; ++i) {
                            for (int j = 0; j < nj; ++j) {
                                 // tmp[i][j] = 0.0;
                                 sum = 0.0;
                                 #pragma omp simd reduction(+:sum)
                                 for (int k = 0; k < nk; ++k)
                                         // tmp[i][j] += alpha * A[i][k] * B[j][k];
                                          sum += alpha * A[i][k] * B[j][k];
                                 tmp[i][j] = sum;  
                            }
                     }
              }
       #pragma omp single
              {
                     #pragma omp taskloop
                     for (int i = 0; i < ni; ++i)
                            for (int j = 0; j < nl; ++j) {
                                   D[i][j] *= beta;
                                   sum = 0.0;
                                   #pragma omp simd reduction(+:sum)
                                   for (int k = 0; k < nj; ++k)
                                          // D[i][j] += tmp[i][k] * C[j][k];
                                          sum += tmp[i][k] * C[j][k];
                                   D[i][j] += sum;
                     }
              }
       }
}


static void init_array_block_task(int ni, int nj, int nk, int nl,
                                         double *alpha,
                                         double *beta,
                                         double tmp[ni][nj],
                                         double A[ni][nk],
                                         double B[nj][nk],
                                         double C[nl][nj],
                                         double D[ni][nl]) {
  *alpha = 1.5;
  *beta = 1.2;
  double div_ni = 1.0 / ni, div_nj = 1.0 / nj, div_nl = 1.0 / nl, div_nk = 1.0 / nk;
  #pragma omp parallel shared(ni, nj, nl, nk, A, B, C, D, div_ni, div_nl, div_nj, div_nk) 
       {
              #pragma omp single
              {
               #pragma omp taskloop
                for (int i = 0; i < ni; ++i)
                  for (int j = 0; j < nk; ++j)
                    A[i][j] = (double)((i*j+1) % ni) * div_ni;

               #pragma omp taskloop
                for (int i = 0; i < nj; ++i)
                  for (int j = 0; j < nk; ++j)
                    B[i][j] = (double)(j*(i+1) % nj) * div_nj;

               #pragma omp taskloop
                for (int i = 0; i < nl; ++i)
                  for (int j = 0; j < nj; ++j)
                    C[i][j] = (double)((j*(i+3)+1) % nl) * div_nl;

               #pragma omp taskloop
                for (int i = 0; i < ni; ++i)
                  for (int j = 0; j < nl; ++j)
                    D[i][j] = (double)(i*(j+2) % nk) * div_nk;

               #pragma omp taskloop
                for (int i = 0; i < ni; ++i)
                  for (int j = 0; j < nj; ++j)
                    tmp[i][j] = 0.0;
              }
       }
}



void mulbl_task(int ni, int nj, int nk, double A[ni][nj], double B[nk][nj], double C[ni][nk], double val)
{
  const int bs = BLOCK_SIZE;

  double a[bs * bs], b[bs * bs], c[bs * bs];
       #pragma omp single 
       {
         for (int bi = 0; bi < ni; bi += bs)  
         // #pragma omp task private(a, b, c) untied firstprivate(bi) 
         {
        #pragma omp taskloop private(a, b, c) firstprivate(bi) 
           for (int bj = 0; bj < nk; bj += bs) 
           {
             for (int p = 0; p < bs; ++p)
               for (int q = 0; q < bs; ++q)
                 c[p * bs + q] = 0.0;

                    for (int bk = 0; bk < nj; bk += bs) 
                    {
                      for (int p = 0; p < bs; ++p)
                        for (int q = 0; q < bs; ++q)
                        {
                          a[p * bs + q] = A[bi + p][bk + q];
                          b[p * bs + q] = B[bj + p][bk + q];
                        }

                      for (int i = 0; i < bs; ++i)
                        for (int j = 0; j < bs; ++j)
                          for (int k = 0; k < bs; ++k)
                            c[i * bs + j] += val * a[i * bs + k] * b[j * bs + k];

                    }
                    for (int p = 0; p < bs; ++p)
                      for (int q = 0; q < bs; ++q)
                        C[bi + p][bj + q] += c[p * bs + q];
            }
          }
       }
}

static void kernel_2mm_block_task(int ni, int nj, int nk, int nl,
                                     double alpha,
                                     double beta,
                                     double tmp[ni][nj],
                                     double A[ni][nk],
                                     double B[nj][nk],
                                     double C[nl][nj],
                                     double D[ni][nl]) {

#pragma omp parallel shared(ni, nj, nl, nk, tmp, A, B, C, D, alpha, beta)
       {

              mulbl_task(ni, nk, nj, A, B, tmp, alpha);

              #pragma omp single
              {
                     #pragma omp taskloop
                       for (int i = 0; i < ni; ++i)
                         for (int j = 0; j < nl; ++j) 
                           D[i][j] *= beta;
              }

              mulbl_task(ni, nj, nl, tmp, C, D, 1.0);

       }
}