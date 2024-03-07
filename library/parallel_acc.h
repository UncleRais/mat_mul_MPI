#include "main.h"
#include <openacc.h>

static void init_array_acc(int ni, int nj, int nk, int nl,
                           double *alpha,
                           double *beta,
                           double A[ni][nk],
                           double B[nj][nk],
                           double C[nl][nj],
                           double D[ni][nl]) {
  *alpha = 1.5;
  *beta = 1.2;
  double div_ni = 1.0 / ni, div_nj = 1.0 / nj, div_nl = 1.0 / nl, div_nk = 1.0 / nk;
  #pragma acc parallel //shared(ni, nj, nl, nk, A, B, C, D, div_ni, div_nl, div_nj, div_nk)
	{
	 #pragma acc loop
	  for (int i = 0; i < ni; ++i)
	    for (int j = 0; j < nk; ++j)
	      A[i][j] = (double)((i*j+1) % ni) * div_ni;

	 #pragma acc loop
	  for (int i = 0; i < nj; ++i)
	    for (int j = 0; j < nk; ++j)
	      B[i][j] = (double)(j*(i+1) % nj) * div_nj;

	 #pragma acc loop
	  for (int i = 0; i < nl; ++i)
	    for (int j = 0; j < nj; ++j)
	      C[i][j] = (double)((j*(i+3)+1) % nl) * div_nl;
	  
	 #pragma acc loop
	  for (int i = 0; i < ni; ++i)
	    for (int j = 0; j < nl; ++j)
	      D[i][j] = (double)(i*(j+2) % nk) * div_nk;
	}
}

static void kernel_2mm_acc(int ni, int nj, int nk, int nl,
                           double alpha,
                           double beta,
                           double tmp[ni][nj],
                           double A[ni][nk],
                           double B[nj][nk],
                           double C[nl][nj],
                           double D[ni][nl]) {

double sum = 0.0;

#pragma acc parallel //shared(ni, nj, nl, nk, tmp, A, B, C, D, alpha, beta, sum)
	{

#pragma acc loop
 	for (int i = 0; i < ni; ++i) {
        for (int j = 0; j < nj; ++j) {
            // tmp[i][j] = 0.0;
            sum = 0.0;
            #pragma acc loop reduction(+:sum)
            for (int k = 0; k < nk; ++k)
                    // tmp[i][j] = tmp[i][j] + alpha * A[i][k] * B[j][k];
            	sum += alpha * A[i][k] * B[j][k];
            tmp[i][j] = sum;
        }
      }

#pragma acc loop
  	for (int i = 0; i < ni; ++i)
        for (int j = 0; j < nl; ++j) {
            D[i][j] *= beta;
            sum = 0.0;
            #pragma acc loop reduction(+:sum)
            for (int k = 0; k < nj; ++k)
                    // D[i][j] = D[i][j] + tmp[i][k] * C[j][k];
            	sum += tmp[i][k] * C[j][k];
            D[i][j] += sum;
        }
    }
}

static void init_array_block_acc(int ni, int nj, int nk, int nl,
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
  #pragma acc parallel //shared(ni, nj, nl, nk, A, B, C, D, div_ni, div_nl, div_nj, div_nk)
	{
	 #pragma acc loop
	  for (int i = 0; i < ni; ++i)
	    for (int j = 0; j < nk; ++j)
	      A[i][j] = (double)((i*j+1) % ni) * div_ni;

	 #pragma acc loop
	  for (int i = 0; i < nj; ++i)
	    for (int j = 0; j < nk; ++j)
	      B[i][j] = (double)(j*(i+1) % nj) * div_nj;

	 #pragma acc loop
	  for (int i = 0; i < nl; ++i)
	    for (int j = 0; j < nj; ++j)
	      C[i][j] = (double)((j*(i+3)+1) % nl) * div_nl;
	  
	 #pragma acc loop
	  for (int i = 0; i < ni; ++i)
	    for (int j = 0; j < nl; ++j)
	      D[i][j] = (double)(i*(j+2) % nk) * div_nk;

	 #pragma acc loop 
	  for (int i = 0; i < ni; ++i)
		for (int j = 0; j < nj; ++j)
		  tmp[i][j] = 0.0;
	}
}


void mulbl_acc(int ni, int nj, int nk, double A[ni][nj], double B[nk][nj], double C[ni][nk], double val)
{
  const int bs = BLOCK_SIZE;

  double a[bs * bs], b[bs * bs], c[bs * bs];

 #pragma acc loop present(a,b,c)
  for (int bi = 0; bi < ni; bi += bs)  //фиксируем номер строки блоков в матрице А
    for (int bj = 0; bj < nk; bj += bs) //фиксируем номер столбца блоков в матрице B (строки блоков в B_T)
    {
      for (int p = 0; p < bs; ++p)
        for (int q = 0; q < bs; ++q)
          c[p * bs + q] = 0.0;

      for (int bk = 0; bk < nj; bk += bs) //идем по строке блоков А и по столбцу блоков В (по строке блоков B_T)
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

static void kernel_2mm_block_acc(int ni, int nj, int nk, int nl,
	                              double alpha,
	                              double beta,
	                              double tmp[ni][nj],
	                              double A[ni][nk],
	                              double B[nj][nk],
	                              double C[nl][nj],
	                              double D[ni][nl]) {

#pragma acc parallel //shared(ni, nj, nl, nk, tmp, A, B, C, D, alpha, beta)
	{

		mulbl_acc(ni, nk, nj, A, B, tmp, alpha);

		#pragma acc loop
		  for (int i = 0; i < ni; ++i)
		    for (int j = 0; j < nl; ++j) 
		      D[i][j] *= beta;

		mulbl_acc(ni, nj, nl, tmp, C, D, 1.0);

	}
}