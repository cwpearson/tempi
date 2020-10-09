

add_executable(bench-mpi-pack bench_mpi_pack.cu ../test/support/type.cpp statistics.cpp)
target_include_directories(bench-mpi-pack PRIVATE ../test/support ../include)
target_link_libraries(bench-mpi-pack PRIVATE tempi::tempi)
target_include_directories(bench-mpi-pack PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(bench-mpi-pack PRIVATE ${MPI_CXX_LIBRARIES})