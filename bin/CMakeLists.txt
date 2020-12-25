add_library(tempi_bin OBJECT benchmark.cpp method.cpp)
tempi_add_output_level_defines(tempi_bin)
target_include_directories(tempi_bin SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(tempi_bin PRIVATE ${MPI_CXX_INCLUDE_DIRS})

macro(add_tempi_executable exe)
    add_executable(${ARGV0} ${ARGN})
    # make tempi support code visible
    target_include_directories(${ARGV0} PRIVATE)
    # link to pmpi first
    target_link_libraries(${ARGV0} PRIVATE tempi::tempi)
    ## just using target_link_libraries(pangolin INTERFACE MPI::MPI_CXX)
    ## causes device linking with -pthread, even as of 3.15.0-rc1
    ## https://gitlab.kitware.com/cmake/cmake/issues/18897
    target_include_directories(${ARGV0} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(${ARGV0} PRIVATE ${MPI_CXX_LIBRARIES})
    target_link_libraries(${ARGV0} PRIVATE tempi_support)
    target_link_libraries(${ARGV0} PRIVATE tempi_bin)
endmacro()

set(TEMPI_BINARY_SOURCES benchmark.cpp)

add_tempi_executable(bench-pack bench_pack.cpp)
add_tempi_executable(bench-send bench_send.cpp)
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
