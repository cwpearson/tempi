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

pmpi_test_exe(internal_type internal_type.cpp)
add_test(NAME internal_type COMMAND ${MPIEXEC_EXECUTABLE} -n 1 internal_type)

pmpi_test_exe(init_finalize init_finalize.cpp)
add_test(NAME init_finalize COMMAND ${MPIEXEC_EXECUTABLE} -n 1 init_finalize)

pmpi_test_exe(isend isend.cu)
add_test(NAME isend COMMAND ${MPIEXEC_EXECUTABLE} -n 1 isend)

pmpi_test_exe(type_commit type_commit.cpp)
add_test(NAME type_commit COMMAND ${MPIEXEC_EXECUTABLE} -n 1 type_commit)