cmake_minimum_required(VERSION 3.22)
project(cuda_solver_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)  # NVIDIA L4 (Ada, sm_89)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set(WARN -Wall -Wextra)
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src/host)

# ---- CPU reference: the same source built serial and with OpenMP ----------
add_executable(cg_serial src/host/cg_cpu.cpp)
target_compile_options(cg_serial PRIVATE ${WARN} -O3 -march=native -Wno-unknown-pragmas)

add_executable(cg_omp src/host/cg_cpu.cpp)
target_compile_options(cg_omp PRIVATE ${WARN} -O3 -march=native)
target_link_libraries(cg_omp PRIVATE OpenMP::OpenMP_CXX)

# ---- CUDA ------------------------------------------------------------------
set(CUDA_FLAGS -O3 -lineinfo -Xcompiler=-fopenmp,-O3,-Wall)

add_executable(cg_cuda src/cuda/cg_cuda.cu)
target_compile_options(cg_cuda PRIVATE ${CUDA_FLAGS})
target_link_libraries(cg_cuda PRIVATE CUDA::cublas CUDA::cusparse OpenMP::OpenMP_CXX)

add_executable(jacobi src/cuda/jacobi.cu)
target_compile_options(jacobi PRIVATE ${CUDA_FLAGS})
target_link_libraries(jacobi PRIVATE OpenMP::OpenMP_CXX)

add_executable(bw_probe src/cuda/bw_probe.cu)
target_compile_options(bw_probe PRIVATE ${CUDA_FLAGS})

# ---- Fortran + OpenACC (gfortran, nvptx offload) ---------------------------
# gfortran 11's nvptx backend emits PTX for sm_35, which the CUDA 12 ptxas
# rejects ("Value 'sm_35' is not defined"). The PTX itself is fine and is
# JIT-compiled by the driver for sm_89, so the assembler's ptxas verification
# step is skipped with -Wa,--no-verify (see docs/DEVLOG.md).
option(BUILD_OPENACC "Build the Fortran OpenACC solver (needs gcc-offload-nvptx)" ON)
if(BUILD_OPENACC)
  enable_language(Fortran)
  set(ACC_FLAGS -fopenacc "-foffload=nvptx-none=-Wa,--no-verify")
  add_executable(cg_acc_gpu src/acc/cg_acc.F90)
  target_compile_options(cg_acc_gpu PRIVATE -O3 ${ACC_FLAGS})
  target_link_options(cg_acc_gpu PRIVATE -O3 ${ACC_FLAGS})
  add_executable(cg_acc_host src/acc/cg_acc.F90)
  target_compile_options(cg_acc_host PRIVATE -O3)
endif()

# ---- tests -------------------------------------------------------------------
enable_testing()
set(T ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME cpu_serial_converges COMMAND cg_serial --n 255 --mode tol --tol 1e-10)
add_test(NAME cpu_openmp_converges COMMAND cg_omp --n 255 --mode tol --tol 1e-10)
add_test(NAME cuda_all_variants_n255 COMMAND cg_cuda --variant all --n 255 --mode tol --tol 1e-10 --ref)
add_test(NAME cuda_all_variants_n512 COMMAND cg_cuda --variant all --n 512 --mode tol --tol 1e-10 --ref)
add_test(NAME jacobi_gpu_matches_cpu COMMAND jacobi --n 257 --check --sweeps 500)
add_test(NAME second_order_convergence COMMAND ${T}/check_order.sh $<TARGET_FILE:cg_cuda>)
if(BUILD_OPENACC)
  add_test(NAME fortran_serial_converges COMMAND cg_acc_host 255 tol 100000 1e-10)
  add_test(NAME fortran_openacc_runs_on_gpu COMMAND ${T}/check_acc_gpu.sh $<TARGET_FILE:cg_acc_gpu>)
endif()
