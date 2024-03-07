#include "utils.h"

namespace init {
constexpr int ni = NI, nk = NK, nj = NJ, nl = NL;

void init_array(double &alpha, double &beta, matrix& A, matrix& B, matrix& C, matrix& D) {
  int i, j;
  alpha = 1.5;
  beta = 1.2;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A(i, j) = (double) ((i*j+1) % ni) / ni;

  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B(i, j) = (double) (i*(j+1) % nj) / nj;

  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C(i, j) = (double) ((i*(j+3)+1) % nl) / nl;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D(i, j) = (double) (i*(j+2) % nk) / nk;
 
}

void kernel_2mm(double alpha, double beta, matrix& tmp, matrix& A, matrix& B, matrix& C, matrix& D) {
 int i, j, k;
 for (i = 0; i < ni; i++)
        for (j = 0; j < nj; j++) {
                tmp(i, j) = 0.0;
                for (k = 0; k < nk; k++)
                        tmp(i, j) += alpha * A(i, k) * B(k, j);
        }

  for (i = 0; i < ni; i++)
        for (j = 0; j < nl; j++) {
                D(i, j) *= beta;
                for (k = 0; k < nj; k++)
                        D(i, j) += tmp(i, k) * C(k, j);
        }
}

}