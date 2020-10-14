
macro(add_tempi_executable exe)
    add_executable(${ARGV0} ${ARGN})
    # make tempi support code visible
    target_include_directories(bench-mpi-pack PRIVATE ../support ../include)
    # link to pmpi first
    target_link_libraries(${ARGV0} PRIVATE tempi::tempi)
    ## just using target_link_libraries(pangolin INTERFACE MPI::MPI_CXX)
    ## causes device linking with -pthread, even as of 3.15.0-rc1
    ## https://gitlab.kitware.com/cmake/cmake/issues/18897
    target_include_directories(${ARGV0} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(${ARGV0} PRIVATE ${MPI_CXX_LIBRARIES})
endmacro()

add_tempi_executable(bench-mpi-pack bench_mpi_pack.cpp ../support/type.cpp statistics.cpp)
add_tempi_executable(bench-mpi-send bench_mpi_send.cpp ../support/type.cpp statistics.cpp)
