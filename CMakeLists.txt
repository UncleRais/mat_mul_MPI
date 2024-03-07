cmake_minimum_required(VERSION 3.16)

project(matrix_multiplication)

find_package(MPI REQUIRED)
if (MPI_BUILD)
    find_package(MPI)
    if(MPI_FOUND)
        add_definitions(-DMPI_BUILD=1)
    endif()
endif()

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(TBB 2021.5.0 REQUIRED)
# set(CMAKE_C_STANDARD 99)
# set(CMAKE_C_STANDARD_REQUIRED on)
# SET(CMAKE_C_COMPILER "/usr/bin/gcc")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -O2")

set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED on)

#SET(CMAKE_CXX_COMPILER "/usr/bin/g++")
#]set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2 ")

include_directories(./library/ ${MPI_INCLUDE_PATH} )
SET(CMAKE_C_COMPILER "/usr/bin/mpicc")
SET(CMAKE_CXX_COMPILER "/usr/bin/mpicxx")
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS} -O2 )
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
set(CMAKE_BUILD_TYPE Release)

#add_subdirectory(src)
set(MPI_PROJ ./src/utils.cpp main_mpi.cpp)	
add_executable(MPI_PROJ ${MPI_PROJ})

# Настраиваем конфигурацию oneTBB - явно отключаем тесты
option(TBB_TEST OFF)

# Подключаем таргет TBB::tbb к приложению
target_link_libraries(MPI_PROJ PRIVATE TBB::tbb)
target_link_libraries(MPI_PROJ PUBLIC MPI::MPI_CXX)


# export OMPI_ALLOW_RUN_AS_ROOT=1
# export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
