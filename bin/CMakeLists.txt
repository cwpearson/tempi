
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
endmacro()

add_tempi_executable(bench-mpi-alltoallv bench_mpi_alltoallv.cpp statistics.cpp)
add_tempi_executable(bench-mpi-pack bench_mpi_pack.cpp ../support/type.cpp statistics.cpp)
add_tempi_executable(bench-mpi-send-types bench_mpi_send_types.cpp ../support/type.cpp statistics.cpp)
add_tempi_executable(bench-mpi-pingpong bench_mpi_pingpong.cpp statistics.cpp)
add_tempi_executable(bench-mpi-isend bench_mpi_isend.cpp statistics.cpp)
add_tempi_executable(bench-halo-exchange bench_halo_exchange.cpp statistics.cpp)

add_tempi_executable(bench-mpi-pattern-random 
  bench_mpi_pattern_random.cpp 
  statistics.cpp 
  ../support/squaremat.cpp
  benchmark.cpp
)

add_tempi_executable(bench-mpi-pattern-blockdiagonal 
  bench_mpi_pattern_blockdiagonal.cpp 
  statistics.cpp 
  ../support/squaremat.cpp
  benchmark.cpp
)

add_tempi_executable(bench-mpi-pattern-permblockdiagonal 
  bench_mpi_pattern_permblockdiagonal.cpp 
  statistics.cpp 
  ../support/squaremat.cpp
  benchmark.cpp
)