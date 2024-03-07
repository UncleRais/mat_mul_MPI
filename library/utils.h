#pragma once
#include "main.h"
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include <omp.h>
#include <mpi.h>
#include <oneapi/tbb.h>

#include <algorithm>
#include <iomanip>
#include <iterator>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <cmath>
#include <cassert>

using T = double;

class matrix {
private:
    std::unique_ptr<T[]> _A;
    int _n, _m;
    int _size;

public:
    matrix(int n, int m, T val = 0.0): _n(n), _m(m), _size(n * m), _A{std::make_unique<T[]>(n * m)} {
        for (int i = 0; i < _size; ++i) _A[i] = val;
    };

    int n() const {return _n;};
    int m() const {return _m;};
    int size() const {return _size;};

    inline T to (int i, int j) const { return _A[i * _m + j]; };
    inline T& operator ()(int i, int j) { return _A[i * _m + j]; };
    inline T& operator [](int i) { return _A[i]; };

    void operator =(const matrix& copy){
        std::copy(copy._A.get(), copy._A.get() + _size, _A.get());
    };

    void save_to_txt(std::string filename) {
      std::ofstream file(filename);
      if (file.is_open()) {
          for (std::size_t i = 0; i < _n; ++i) {
            for (std::size_t j = 0; j < _m; ++j) 
              file << std::setw(9) << std::setprecision(6) << (*this)(i, j) << " ";
            file << "\n";
          };
          file.close(); 
          //std::cout << "File has been written." << std::endl;
      } else {
        std::cout << "Error. File wasn't opened." << std::endl;
      };
      
    };
};

void print_array(const matrix& D) {
  std::cout << "==BEGIN DUMP_ARRAYS==" << std::endl;
  std::cout << "begin dump:" << std::endl;
  const int cols = std::min(D.n(), 5), rows = std::min(D.m(), 5);
  for (int i = 0; i < cols; ++i) {
    for (int j = 0; j < rows; ++j) {
  		std::cout << D.to(i, j) << " ";
  	}
    std::cout << std::endl;
  }
  std::cout << "\nend   dump:" << std::endl;
  std::cout << "==END   DUMP_ARRAYS==" << std::endl;
};

double matrix_equal(const matrix& tmp1, const matrix& tmp2) {
  const int n = tmp1.n(), m = tmp1.m();
  double norm = 0.0;
  for (std::size_t i = 0; i < n; ++i) 
    for (std::size_t j = 0; j < m; ++j) 
      norm += std::fabs(tmp1.to(i, j) - tmp2.to(i, j));
  norm /= n * m;
  return(norm);
};

void file_printf(FILE* file, char* message, char* str_data, double num_data) {
  if (fabs(num_data + 1.0) < 1e-5) {
  	sprintf(message, "%s %s", str_data, "\n");
  }
  else {
  	sprintf(message, "%s %f %s", str_data, num_data, "\n");
  }
  fputs(message, file);
};