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

# ---- Fortran + OpenACC / do concurrent (nvfortran, NVIDIA HPC SDK) -----------
# Optional: built only when nvfortran is found (PATH, NVHPC_ROOT, or the
# trimmed tarball unpack in ~/opt/nvhpc, see docs/DEVLOG.md). CMake's Fortran
# language is gfortran here, so nvfortran is driven by custom commands. The
# SDK's bundled CUDA was not unpacked; NVHPC_CUDA_HOME points nvfortran at the
# same CUDA toolkit nvcc uses. -Minfo output lands next to the binaries.
option(BUILD_NVFORTRAN "Build the nvfortran OpenACC / stdpar solvers if nvfortran is found" ON)
if(BUILD_NVFORTRAN)
  find_program(NVFORTRAN nvfortran
    HINTS $ENV{NVHPC_ROOT}/compilers/bin
          $ENV{HOME}/opt/nvhpc/install_components/Linux_x86_64/26.9/compilers/bin)
endif()
if(BUILD_NVFORTRAN AND NVFORTRAN)
  message(STATUS "nvfortran: ${NVFORTRAN}")
  get_filename_component(NVF_CUDA_HOME "${CUDAToolkit_BIN_DIR}" DIRECTORY)
  set(NVF ${CMAKE_COMMAND} -E env NVHPC_CUDA_HOME=${NVF_CUDA_HOME} ${NVFORTRAN})
  set(ACC_SRC ${CMAKE_SOURCE_DIR}/src/acc/cg_acc.F90)
  add_custom_command(OUTPUT cg_acc_nvf cg_acc_nvf.minfo.txt
    COMMAND ${NVF} -O3 -acc=gpu -gpu=cc89 -Minfo=accel ${ACC_SRC} -o cg_acc_nvf > cg_acc_nvf.minfo.txt 2>&1
    DEPENDS ${ACC_SRC} COMMENT "nvfortran -acc=gpu cg_acc.F90" VERBATIM)
  # do concurrent build: -acc=gpu is only there for acc_get_property_string /
  # acc_init; the OpenACC data region is compiled out by -DUSE_DC.
  add_custom_command(OUTPUT cg_dc_nvf cg_dc_nvf.minfo.txt
    COMMAND ${NVF} -O3 -DUSE_DC -stdpar=gpu -acc=gpu -gpu=cc89 -Minfo=accel,stdpar ${ACC_SRC} -o cg_dc_nvf > cg_dc_nvf.minfo.txt 2>&1
    DEPENDS ${ACC_SRC} COMMENT "nvfortran -stdpar=gpu cg_acc.F90 (do concurrent)" VERBATIM)
  add_custom_target(nvfortran_solvers ALL DEPENDS cg_acc_nvf cg_dc_nvf)
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
if(BUILD_NVFORTRAN AND NVFORTRAN)
  add_test(NAME nvfortran_openacc_runs_on_gpu COMMAND ${T}/check_nvf_gpu.sh ${CMAKE_BINARY_DIR}/cg_acc_nvf)
  add_test(NAME nvfortran_do_concurrent_runs_on_gpu COMMAND ${T}/check_nvf_gpu.sh ${CMAKE_BINARY_DIR}/cg_dc_nvf)
endif()
