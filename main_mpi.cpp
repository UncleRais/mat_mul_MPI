#include "initial.h"
#include "initial_optimized.h"
#include "block_multiplication.h"
#include "mpi_impl.h"
#include "clock.h"

int main(int argc, char** argv) {

int rank, size, DRAGON_WARRIOR = 0;
bool verbose = false;

if (rank == DRAGON_WARRIOR && verbose) {
std::cout << "TBB version: " << TBB_runtime_version() << std::endl;
};

MPI_Init(&argc, &argv);

mpi_timer_start();
mpi_timer_stop();

MPI_Comm_size(MPI_COMM_WORLD, &size);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);

if (rank == DRAGON_WARRIOR) {
time_t mytime = time(NULL);
struct tm *now = localtime(&mytime);
std::cout << "Date: " << now->tm_mday << "." << now->tm_mon + 1 << "." << now->tm_year + 1900 << "  ";
std::cout << "Time: " << now->tm_hour << ":" << now->tm_min << ":" << now->tm_sec << std::endl;
};

//////// SEQUENTIAL
if (rank == DRAGON_WARRIOR) {
std::cout << "-----------------------------SEQUENTIAL-----------------------------" << std::endl;
constexpr int ni = NI, nk = NK, nj = NJ, nl = NL;
constexpr int iterations = 1;
double alpha;
double beta;
matrix tmp(ni, nj); matrix tmp_ref(ni, nj); 
matrix A(ni, nk);
matrix B(nk, nj); matrix B_T(nj, nk); 
matrix C(nj, nl); matrix C_T(nl, nj); 
matrix D(ni, nl); matrix D_ref(ni, nl);

std::array<double, 3> times; 

std::cout << "Starting initial calculation." << std::endl;
for (int iter = 0; iter < iterations; ++iter) {
  mpi_timer_start();

  init::init_array(alpha, beta, A, B, C, D);

  init::kernel_2mm(alpha, beta, tmp_ref, A, B, C, D);

  mpi_timer_stop();
  times[0] += bench_timer_print();
}
times[0] = times[0] / iterations;
std::cout << "Sum(times) / iterations = mean_time = " << times[0] << std::endl;
std::cout << "Initial calculation is finished." << std::endl;
D_ref = D;

//////// OPTIMIZED
std::cout << "Starting sequential optimized calculation." << std::endl;
for (int iter = 0; iter < iterations; ++iter) {
  mpi_timer_start();

  opt::init_array(alpha, beta, A, B_T, C_T, D);

  opt::kernel_2mm(alpha, beta, tmp, A, B_T, C_T, D);
  
  mpi_timer_stop();
  times[1] += bench_timer_print();
}
times[1] = times[1] / iterations;
std::cout << "Sum(times) / iterations = mean_time = " <<  times[1] << std::endl;
std::cout << "Sequential optimized calculation is finished." << std::endl;
std::cout << "Sum_{i,j}|tmp_optimized[i][j] - tmp_sequential[i][j]| = " << matrix_equal(tmp, tmp_ref)<< std::endl;
std::cout << "Sum_{i,j}|D_optimized[i][j] - D_sequential[i][j]| = " << matrix_equal(D, D_ref)<< std::endl;
////////

//////// BLOCK SEQUENTIAL
std::cout << "Starting sequential block_mul calculation." << std::endl;
for (int iter = 0; iter < iterations; ++iter) {
  mpi_timer_start();

  block::init_array(alpha, beta, tmp, A, B_T, C_T, D);

  block::kernel_2mm(alpha, beta, tmp, A, B_T, C_T, D);
  
  mpi_timer_stop();
  times[2] += bench_timer_print();
}
times[2] = times[2] / iterations;
std::cout << "Sum(times) / iterations = mean_time = " <<  times[2] << std::endl;
std::cout << "Sequential optimized calculation is finished." << std::endl;
std::cout << "Sum_{i,j}|tmp_block[i][j] - tmp_sequential[i][j]| = " <<  matrix_equal(tmp, tmp_ref) << std::endl;
std::cout << "Sum_{i,j}|D_block[i][j] - D_sequential[i][j]| = " <<  matrix_equal(D, D_ref) << std::endl;
////////

std::cout << "Initial / Optimized: " << times[0] / times[1] << std::endl;
std::cout << "Initial / Block: " << times[0] / times[2] << std::endl;
if (verbose) {
  D_ref.save_to_txt("D_ref.txt");
  tmp_ref.save_to_txt("tmp_ref.txt");
};
std::cout << "-----------------------------SEQUENTIAL-----------------------------" << std::endl;
};


MPI_Barrier(MPI_COMM_WORLD);
//////// MPI

if (rank == DRAGON_WARRIOR) std::cout << "-----------------------------MPI-----------------------------" << std::endl;
  
constexpr int ni = NI, nk = NK, nj = NJ, nl = NL;
assert(ni % size == 0);
const int ni_pr = ni / size;
constexpr int iterations = 3;
double alpha;
double beta;
matrix tmp(ni_pr, nj);
matrix A(ni_pr, nk);
matrix B(nk, nj); matrix B_T(nj, nk); 
matrix C(nj, nl); matrix C_T(nl, nj); 
matrix D(ni_pr, nl); 

double times[3]; 


if (rank == DRAGON_WARRIOR) std::cout << "Starting initial calculation." << std::endl;
for (int iter = 0; iter < iterations; ++iter) {
  mpi_timer_start();

  mpi::init::init_array(rank, size, alpha, beta, A, B, C, D);

  mpi::init::kernel_2mm(rank, size, alpha, beta, tmp, A, B, C, D);

  mpi_timer_stop();
  times[0] += bench_timer_print();
}
times[0] /= iterations;
double time;
// std::cout << "Process №" << rank << " finished." << std::endl;
MPI_Reduce(&(times[0]), &time, 1, MPI_DOUBLE, MPI_SUM, DRAGON_WARRIOR, MPI_COMM_WORLD);
if (rank == DRAGON_WARRIOR) time /= size;
if (rank == DRAGON_WARRIOR) std::cout << "Sum(times) / iterations = mean_time = " << time << std::endl;
if (rank == DRAGON_WARRIOR) std::cout << "Initial calculation is finished." << std::endl;

MPI_Barrier(MPI_COMM_WORLD);

if (rank == DRAGON_WARRIOR) std::cout << "Starting optimized calculation." << std::endl;
for (int iter = 0; iter < iterations; ++iter) {
  mpi_timer_start();

  mpi::opt::init_array(rank, size, alpha, beta, A, B_T, C_T, D);

  mpi::opt::kernel_2mm(rank, size, alpha, beta, tmp, A, B_T, C_T, D);

  mpi_timer_stop();
  times[1] += bench_timer_print();
}
times[1] /= iterations;
// std::cout << "Process №" << rank << " finished." << std::endl;
MPI_Reduce(&(times[1]), &time, 1, MPI_DOUBLE, MPI_SUM, DRAGON_WARRIOR, MPI_COMM_WORLD);
if (rank == DRAGON_WARRIOR) time /= size;
if (rank == DRAGON_WARRIOR) std::cout << "Sum(times) / iterations = mean_time = " << time << std::endl;
if (rank == DRAGON_WARRIOR) std::cout << "Optimized calculation is finished." << std::endl;

MPI_Barrier(MPI_COMM_WORLD);

if (rank == DRAGON_WARRIOR) std::cout << "Starting block calculation." << std::endl;
for (int iter = 0; iter < iterations; ++iter) {
  mpi_timer_start();

  mpi::block::init_array(rank, size, alpha, beta, tmp, A, B_T, C_T, D);

  mpi::block::kernel_2mm(rank, size, alpha, beta, tmp, A, B_T, C_T, D);

  mpi_timer_stop();
  times[2] += bench_timer_print();
}
times[2] /= iterations;
// std::cout << "Process №" << rank << " finished." << std::endl;
MPI_Reduce(&(times[2]), &time, 1, MPI_DOUBLE, MPI_SUM, DRAGON_WARRIOR, MPI_COMM_WORLD);
if (rank == DRAGON_WARRIOR) time /= size;
if (rank == DRAGON_WARRIOR) std::cout << "Sum(times) / iterations = mean_time = " << time << std::endl;
if (rank == DRAGON_WARRIOR) std::cout << "Block calculation is finished. \n" << std::endl;

if (rank == DRAGON_WARRIOR) std::cout << "Initial / Optimized: " << times[0] / times[1] << std::endl;
if (rank == DRAGON_WARRIOR) std::cout << "Initial / Block: " << times[0] / times[2] << std::endl;

MPI_Barrier(MPI_COMM_WORLD);

if (verbose) {
  mpi_timer_start();
  D.save_to_txt(std::string("D_") + std::to_string(rank) + std::string(".txt"));
  tmp.save_to_txt(std::string("tmp_") + std::to_string(rank) + std::string(".txt"));
  mpi_timer_stop();
  std::cout << "[" << rank << "]: " << "Time of matrix writing = " << bench_timer_print() << std::endl;
};


MPI_Barrier(MPI_COMM_WORLD);
if (rank == DRAGON_WARRIOR) std::cout << "-----------------------------MPI-----------------------------" << std::endl;


MPI_Finalize();

return 0;
}
