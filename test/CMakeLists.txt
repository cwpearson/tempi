macro(pmpi_test_exe exe)
    add_executable(${ARGV0} ${ARGN})
    # link to pmpi first
    target_link_libraries(${ARGV0} PRIVATE tempi::tempi)
    ## just using target_link_libraries(pangolin INTERFACE MPI::MPI_CXX)
    ## causes device linking with -pthread, even as of 3.15.0-rc1
    ## https://gitlab.kitware.com/cmake/cmake/issues/18897
    target_include_directories(${ARGV0} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(${ARGV0} PRIVATE ${MPI_CXX_LIBRARIES})

    target_link_libraries(${ARGV0} PRIVATE tempi_support)

    if(TEMPI_ENABLE_METIS)
      target_compile_definitions(${ARGV0} PRIVATE TEMPI_ENABLE_METIS)
    endif()
    if(TEMPI_ENABLE_KAHIP)
      target_compile_definitions(${ARGV0} PRIVATE TEMPI_ENABLE_KAHIP)
    endif()


endmacro()


# some tests just check semantics of MPI implementation
pmpi_test_exe(type_size type_size.cpp)
add_test(NAME type_size COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 type_size)

# tests for tempi functionality
pmpi_test_exe(init_finalize init_finalize.cpp)
add_test(NAME init_finalize COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 init_finalize)

pmpi_test_exe(partition_kahip partition_kahip.cpp)
add_test(NAME partition_kahip COMMAND partition_kahip)

# KaHIP process mapping often returns unbalanced partitions
# pmpi_test_exe(partition_kahip_process_mapping partition_kahip_process_mapping.cpp)
# add_test(NAME partition_kahip_process_mapping COMMAND partition_kahip_process_mapping)

pmpi_test_exe(type_commit type_commit.cpp)
add_test(NAME type_commit COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 type_commit)

pmpi_test_exe(pack_unpack pack_unpack.cpp)
add_test(NAME pack_unpack COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 pack_unpack)

pmpi_test_exe(isend isend.cu)
add_test(NAME isend COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 isend)

pmpi_test_exe(send send.cpp)
add_test(NAME send COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 send)

pmpi_test_exe(dist_graph_create_adjacent dist_graph_create_adjacent.cpp)
add_test(NAME dist_graph_create_adjacent COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 send)