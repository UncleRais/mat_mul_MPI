#include "utils.h"

namespace opt {
constexpr int ni = NI, nk = NK, nj = NJ, nl = NL;

void init_array(double &alpha, double &beta, matrix& A, matrix& B, matrix& C, matrix& D) {

  alpha = 1.5;
  beta = 1.2;
  double div_ni = 1.0 / ni, div_nj = 1.0 / nj, div_nl = 1.0 / nl, div_nk = 1.0 / nk;
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nk; ++j)
      A(i, j) = (double)((i*j+1) % ni) * div_ni;

  for (int i = 0; i < nj; ++i)
    for (int j = 0; j < nk; ++j)
      B(i, j) = (double)(j*(i+1) % nj) * div_nj;

  for (int i = 0; i < nl; ++i)
    for (int j = 0; j < nj; ++j)
      C(i, j) = (double)((j*(i+3)+1) % nl) * div_nl;
 
  for (int i = 0; i < ni; ++i)
    for (int j = 0; j < nl; ++j)
      D(i, j) = (double)(i*(j+2) % nk) * div_nk;

}

void kernel_2mm(double alpha, double beta, matrix& tmp, matrix& A, matrix& B, matrix& C, matrix& D) {

 for (int i = 0; i < ni; ++i) {
        for (int j = 0; j < nj; ++j) {
                tmp(i, j) = 0.0;
                for (int k = 0; k < nk; ++k)
                        tmp(i, j) += alpha * A(i, k) * B(j, k);
        }
      }

  for (int i = 0; i < ni; ++i)
        for (int j = 0; j < nl; ++j) {
                D(i, j) *= beta;
                for (int k = 0; k < nj; ++k)
                        D(i, j) += tmp(i, k) * C(j, k);
        }
}
}