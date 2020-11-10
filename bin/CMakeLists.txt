add_library(tempi_bin OBJECT benchmark.cpp method.cpp statistics.cpp)
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


add_tempi_executable(bench-mpi-pack bench_mpi_pack.cpp)
add_tempi_executable(bench-mpi-send-types bench_mpi_send_types.cpp)
add_tempi_executable(bench-mpi-pingpong bench_mpi_pingpong.cpp)
add_tempi_executable(bench-mpi-isend bench_mpi_isend.cpp)
add_tempi_executable(bench-halo-exchange bench_halo_exchange.cpp)

add_tempi_executable(bench-mpi-alltoallv bench_mpi_alltoallv.cpp)
add_tempi_executable(bench-mpi-random-alltoallv bench_mpi_random_alltoallv.cpp)
add_tempi_executable(bench-mpi-random-isend-irecv bench_mpi_random_isend_irecv.cpp)
add_tempi_executable(bench-mpi-random-sparse-isend-irecv bench_mpi_random_sparse_isend_irecv.cpp)
add_tempi_executable(bench-mpi-random-neighbor-alltoallv bench_mpi_random_neighbor_alltoallv.cpp)

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