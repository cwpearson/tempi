##          Copyright Carl Pearson 2020 - 2021.
## Distributed under the Boost Software License, Version 1.0.
##    (See accompanying file LICENSE or copy at
##    https://www.boost.org/LICENSE_1_0.txt)


add_library(tempi_bin OBJECT benchmark.cpp method.cpp)
tempi_add_output_level_defines(tempi_bin)
target_include_directories(tempi_bin SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(tempi_bin PRIVATE ${MPI_CXX_INCLUDE_DIRS})

macro(add_tempi_executable exe)
    add_executable(${ARGV0} ${ARGN})
    # link to pmpi first
    target_link_libraries(${ARGV0} PRIVATE tempi::tempi)
    ## just using target_link_libraries(pangolin INTERFACE MPI::MPI_CXX)
    ## causes device linking with -pthread, even as of 3.15.0-rc1
    ## https://gitlab.kitware.com/cmake/cmake/issues/18897
    target_include_directories(${ARGV0} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(${ARGV0} PRIVATE ${MPI_CXX_LIBRARIES})
    target_link_libraries(${ARGV0} PRIVATE tempi_support)
    target_link_libraries(${ARGV0} PRIVATE tempi_bin)

    # add thirdparty for nonstd::optional
    target_include_directories(${ARGV0} PRIVATE SYSTEM ../thirdparty)

  # prefer march=native, but fallback to mcpu=native if needed. march implies mtune
  if (CXX_HAS_MARCH)
    message(STATUS "-march=native detected")
    target_compile_options(
      ${ARGV0}
      PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>
      $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    )
  elseif (CXX_HAS_MCPU)
    message(STATUS "-mcpu=native -mtune=native instead of -march=native")
    target_compile_options(
      ${ARGV0}
      PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mcpu=native -Xcompiler=-mtune=native>
      $<$<COMPILE_LANGUAGE:CXX>:-mcpu=native -mtune=native>
    )
  endif()
endmacro()

add_tempi_executable(bench-pack bench_pack.cpp)
add_tempi_executable(bench-cache bench_cache.cpp)
add_tempi_executable(measure-system measure_system.cpp)
add_tempi_executable(bench-pack-kernels bench_pack_kernels.cu)
add_tempi_executable(bench-mpi-pack bench_mpi_pack.cpp)
add_tempi_executable(bench-mpi-pingpong-1d bench_mpi_pingpong_1d.cpp)
add_tempi_executable(bench-mpi-pingpong-nd bench_mpi_pingpong_nd.cpp)
add_tempi_executable(bench-mpi-isend bench_mpi_isend.cpp)
add_tempi_executable(bench-halo-exchange bench_halo_exchange.cpp)
add_tempi_executable(bench-mpi-ireduce bench_mpi_ireduce.cpp)

add_tempi_executable(bench-type-commit bench_type_commit.cpp)
add_tempi_executable(bench-alltoallv-random-sparse bench_alltoallv_random_sparse.cpp)
add_tempi_executable(bench-nbr-alltoallv-random-sparse bench_nbr_alltoallv_random_sparse.cpp)

#add_tempi_executable(bench-mpi-pattern-blockdiagonal 
#  bench_mpi_pattern_blockdiagonal.cpp 
#  statistics.cpp 
#  ../support/squaremat.cpp
#  method.cpp
#  benchmark.cpp
#)

# add_tempi_executable(bench-mpi-pattern-permblockdiagonal 
#   bench_mpi_pattern_permblockdiagonal.cpp 
#   statistics.cpp 
#   ../support/squaremat.cpp
#   method.cpp
#   benchmark.cpp
# )
