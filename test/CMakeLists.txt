find_package(MPI REQUIRED)
message(STATUS "MPI_CXX_LIBRARIES=" ${MPI_CXX_LIBRARIES})
message(STATUS "MPIEXEC_EXECUTABLE=" ${MPIEXEC_EXECUTABLE})

macro(pmpi_test_exe exe)
    add_executable(${ARGV0} ${ARGN})
    # link to pmpi first
    target_link_libraries(${ARGV0} PRIVATE pmpi)
    ## just using target_link_libraries(pangolin INTERFACE MPI::MPI_CXX)
    ## causes device linking with -pthread, even as of 3.15.0-rc1
    ## https://gitlab.kitware.com/cmake/cmake/issues/18897
    target_include_directories(${ARGV0} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(${ARGV0} PRIVATE ${MPI_CXX_LIBRARIES})
endmacro()

add_executable(init_finalize init_finalize.cpp)
# link to pmpi first
target_link_libraries(init_finalize PRIVATE pmpi)
## just using target_link_libraries(pangolin INTERFACE MPI::MPI_CXX)
## causes device linking with -pthread, even as of 3.15.0-rc1
## https://gitlab.kitware.com/cmake/cmake/issues/18897
target_include_directories(init_finalize PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_link_libraries(init_finalize PRIVATE ${MPI_CXX_LIBRARIES})
add_test(NAME init_finalize COMMAND ${MPIEXEC_EXECUTABLE} -n 1 init_finalize)

pmpi_test_exe(isend isend.cu)
add_test(NAME isend COMMAND ${MPIEXEC_EXECUTABLE} -n 1 init_finalize)