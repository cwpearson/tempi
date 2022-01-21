##          Copyright Carl Pearson 2020 - 2021.
## Distributed under the Boost Software License, Version 1.0.
##    (See accompanying file LICENSE or copy at
##    https://www.boost.org/LICENSE_1_0.txt)

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

    # add thirdparty for nonstd::optional
    target_include_directories(${ARGV0} PRIVATE SYSTEM ../thirdparty)

    if(TEMPI_ENABLE_METIS)
      target_compile_definitions(${ARGV0} PRIVATE TEMPI_ENABLE_METIS)
    endif()
    if(TEMPI_ENABLE_KAHIP)
      target_compile_definitions(${ARGV0} PRIVATE TEMPI_ENABLE_KAHIP)
    endif()
endmacro()

pmpi_test_exe(iid iid.cpp)
add_test(NAME iid COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/iid)

pmpi_test_exe(numeric numeric.cpp)
add_test(NAME numeric COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/numeric)

pmpi_test_exe(measure_system measure_system.cpp)
add_test(NAME measure_system COMMAND measure_system)

pmpi_test_exe(type_equivalence type_equivalence.cpp)
add_test(NAME type_equivalence COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/type_equivalence)


pmpi_test_exe(init_finalize init_finalize.cpp)
add_test(NAME init_finalize COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/init_finalize)

if(TEMPI_ENABLE_KAHIP)
pmpi_test_exe(partition_kahip partition_kahip.cpp)
add_test(NAME partition_kahip COMMAND ${CMAKE_CURRENT_BINARY_DIR}/partition_kahip)

# KaHIP process mapping often returns unbalanced partitions
pmpi_test_exe(partition_kahip_process_mapping partition_kahip_process_mapping.cpp)
add_test(NAME partition_kahip_process_mapping COMMAND ${CMAKE_CURRENT_BINARY_DIR}/partition_kahip_process_mapping)
endif()

pmpi_test_exe(type_commit type_commit.cpp)
add_test(NAME type_commit COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/type_commit)

pmpi_test_exe(pack_unpack pack_unpack.cpp)
add_test(NAME pack_unpack COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/pack_unpack)

pmpi_test_exe(isend isend.cu)
add_test(NAME isend COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/isend)

pmpi_test_exe(isend_contiguous isend_contiguous.cu)
add_test(NAME isend_contiguous COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${CMAKE_CURRENT_BINARY_DIR}/isend_contiguous)

pmpi_test_exe(sender sender.cpp)
add_test(NAME sender COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/sender)

pmpi_test_exe(send send.cpp)
add_test(NAME send COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/send)

pmpi_test_exe(send_vector send_vector.cpp)
add_test(NAME send_vector COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${CMAKE_CURRENT_BINARY_DIR}/send_vector)

pmpi_test_exe(dist_graph_create_adjacent dist_graph_create_adjacent.cpp)
add_test(NAME dist_graph_create_adjacent COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${CMAKE_CURRENT_BINARY_DIR}/dist_graph_create_adjacent)
