#include "utils.h"

namespace mpi {
constexpr int ni = NI, nk = NK, nj = NJ, nl = NL;


namespace init{

void init_array(int rank, int size, double &alpha, double &beta, matrix& A, matrix& B, matrix& C, matrix& D) {
  const int ni_pr = ni / size;
  int i, j;
  alpha = 1.5;
  beta = 1.2;
  for (i = 0; i < ni_pr; i++)
    for (j = 0; j < nk; j++)
      A(i, j) = (double) (((i + rank * ni_pr)*j+1) % ni) / ni;

  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B(i, j) = (double) (i*(j+1) % nj) / nj;

  for (i = 0; i < nj; i++)
    for (j = 0; j < nl; j++)
      C(i, j) = (double) ((i*(j+3)+1) % nl) / nl;

  for (i = 0; i < ni_pr; i++)
    for (j = 0; j < nl; j++)
      D(i, j) = (double) ((i + rank * ni_pr)*(j+2) % nk) / nk;
 
}

void kernel_2mm(int rank, int size, double alpha, double beta, matrix& tmp, matrix& A, matrix& B, matrix& C, matrix& D) {
const int ni_pr = ni / size;
 int i, j, k;
 for (i = 0; i < ni_pr; i++)
        for (j = 0; j < nj; j++) {
                tmp(i, j) = 0.0;
                for (k = 0; k < nk; k++)
                        tmp(i, j) += alpha * A(i, k) * B(k, j);
        }

  for (i = 0; i < ni_pr; i++)
        for (j = 0; j < nl; j++) {
                D(i, j) *= beta;
                for (k = 0; k < nj; k++)
                        D(i, j) += tmp(i, k) * C(k, j);
        }
}
}


namespace opt{
void init_array(int rank, int size, double &alpha, double &beta, matrix& A, matrix& B, matrix& C, matrix& D) {
  const int ni_pr = ni / size;
  alpha = 1.5;
  beta = 1.2;
  double div_ni = 1.0 / ni, div_nj = 1.0 / nj, div_nl = 1.0 / nl, div_nk = 1.0 / nk;
  for (int i = 0; i < ni_pr; ++i)
    for (int j = 0; j < nk; ++j)
      A(i, j) = (double)(((i + rank * ni_pr)*j+1) % ni) * div_ni;

  for (int i = 0; i < nj; ++i)
    for (int j = 0; j < nk; ++j)
      B(i, j) = (double)(j*(i+1) % nj) * div_nj;

  for (int i = 0; i < nl; ++i)
    for (int j = 0; j < nj; ++j)
      C(i, j) = (double)((j*(i+3)+1) % nl) * div_nl;
 
  for (int i = 0; i < ni_pr; ++i)
    for (int j = 0; j < nl; ++j)
      D(i, j) = (double)((i + rank * ni_pr)*(j+2) % nk) * div_nk;

}

void kernel_2mm(int rank, int size, double alpha, double beta, matrix& tmp, matrix& A, matrix& B, matrix& C, matrix& D) {
const int ni_pr = ni / size;
 for (int i = 0; i < ni_pr; ++i) {
        for (int j = 0; j < nj; ++j) {
                tmp(i, j) = 0.0;
                for (int k = 0; k < nk; ++k)
                        tmp(i, j) += alpha * A(i, k) * B(j, k);
        }
      }

  for (int i = 0; i < ni_pr; ++i)
        for (int j = 0; j < nl; ++j) {
                D(i, j) *= beta;
                for (int k = 0; k < nj; ++k)
                        D(i, j) += tmp(i, k) * C(j, k);
        }
}
}

namespace block{
void init_array(int rank, int size, double &alpha, double &beta, matrix& tmp, matrix& A, matrix& B, matrix& C, matrix& D) {
  const int ni_pr = ni / size;
  alpha = 1.5;
  beta = 1.2;
  double div_ni = 1.0 / ni, div_nj = 1.0 / nj, div_nl = 1.0 / nl, div_nk = 1.0 / nk;
  for (int i = 0; i < ni_pr; ++i)
    for (int j = 0; j < nk; ++j)
      A(i, j) = (double)(((i + rank * ni_pr)*j+1) % ni) * div_ni;

  for (int i = 0; i < nj; ++i)
    for (int j = 0; j < nk; ++j)
      B(i, j) = (double)(j*(i+1) % nj) * div_nj;

  for (int i = 0; i < nl; ++i)
    for (int j = 0; j < nj; ++j)
      C(i, j) = (double)((j*(i+3)+1) % nl) * div_nl;

  for (int i = 0; i < ni_pr; ++i)
    for (int j = 0; j < nl; ++j)
      D(i, j) = (double)((i + rank * ni_pr)*(j+2) % nk) * div_nk;

  for (int i = 0; i < ni_pr; ++i)
    for (int j = 0; j < nj; ++j)
      tmp(i, j) = 0.0;
}

void mulbl(int ni, int nj, int nk, matrix& A, matrix& B, matrix& C, double val)
{
  const int bs = BLOCK_SIZE;

  double a[bs * bs], b[bs * bs], c[bs * bs];

  for (int bi = 0; bi < ni; bi += bs) //фиксируем номер строки блоков в матрице А
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
            a[p * bs + q] = A(bi + p, bk + q);
            b[p * bs + q] = B(bj + p, bk + q);
          }

        for (int i = 0; i < bs; ++i)
          for (int j = 0; j < bs; ++j)
            for (int k = 0; k < bs; ++k)
              c[i * bs + j] += val * a[i * bs + k] * b[j * bs + k];

      }
      for (int p = 0; p < bs; ++p)
        for (int q = 0; q < bs; ++q)
          C(bi + p, bj + q) += c[p * bs + q];
    }
}

void kernel_2mm(int rank, int size, double alpha, double beta, matrix& tmp, matrix& A, matrix& B, matrix& C, matrix& D) {
  const int ni_pr = ni / size;
  mulbl(ni_pr, nk, nj, A, B, tmp, alpha);

  for (int i = 0; i < ni_pr; ++i)
    for (int j = 0; j < nl; ++j) 
      D(i, j) *= beta;

  mulbl(ni_pr, nj, nl, tmp, C, D, 1.0);
}
}


}