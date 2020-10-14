find_package(MPI REQUIRED)
message(STATUS "MPI_CXX_LIBRARIES=" ${MPI_CXX_LIBRARIES})
message(STATUS "MPIEXEC_EXECUTABLE=" ${MPIEXEC_EXECUTABLE})

macro(pmpi_test_exe exe)
    add_executable(${ARGV0} ${ARGN} ../support/type.cpp)
    # link to pmpi first
    target_link_libraries(${ARGV0} PRIVATE tempi::tempi)
    ## just using target_link_libraries(pangolin INTERFACE MPI::MPI_CXX)
    ## causes device linking with -pthread, even as of 3.15.0-rc1
    ## https://gitlab.kitware.com/cmake/cmake/issues/18897
    target_include_directories(${ARGV0} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(${ARGV0} PRIVATE ${MPI_CXX_LIBRARIES})
endmacro()


# some tests just check semantics of MPI implementation
pmpi_test_exe(type_size type_size.cpp)
add_test(NAME type_size COMMAND ${MPIEXEC_EXECUTABLE} -n 1 type_size)

# tests for tempi functionality
pmpi_test_exe(init_finalize init_finalize.cpp)
add_test(NAME init_finalize COMMAND ${MPIEXEC_EXECUTABLE} -n 1 init_finalize)

pmpi_test_exe(isend isend.cu)
add_test(NAME isend COMMAND ${MPIEXEC_EXECUTABLE} -n 1 isend)

pmpi_test_exe(type_commit type_commit.cpp)
add_test(NAME type_commit COMMAND ${MPIEXEC_EXECUTABLE} -n 1 type_commit)

pmpi_test_exe(pack pack.cpp)
add_test(NAME pack COMMAND ${MPIEXEC_EXECUTABLE} -n 1 pack)