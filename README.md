# mat_mul_MPI

Contains sequential and parallel implementations of following algorithms:

* General matrix multiplication
* Block matrix multiplication

Note that present implementations are intended for academic purposes, as such they are not meant to be used in any sort of high-performance production code.

## Compilation

* Recommended compilers: mpirun
* Requires -std=c++17
* Requires oneTBB 2021.11.0

* ## Usage

Solving task:<br>
$$T = \alpha \cdot A \cdot B, \quad D = \beta \cdot D + T \cdot C, \quad \alpha, \beta \in \mathbb{R}, $$
where<br>
$T_{n_i \times n_j} , A_{n_i \times n_k} , B_{n_k \times n_j} , C_{n_j \times n_l}, D_{n_i \times n_l}$ - real matrices, $n_i , n_j , n_l , n_k \in \mathbb{N}$.<br>

## Example

Compilation of "main_mpi.cpp" with level of optimization -O2 in Release (default). The result of code's work is the mean time (sum / number_of_iterations) of algorithms' work.
```bash
mkdir build
cd build
cmake ..
make -j8
mpirun -np 10 ./MPI_PROJ 
```
  
