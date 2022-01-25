##          Copyright Carl Pearson 2020 - 2021.
## Distributed under the Boost Software License, Version 1.0.
##    (See accompanying file LICENSE or copy at
##    https://www.boost.org/LICENSE_1_0.txt)

macro(tempi_test_exe exe)
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

tempi_test_exe(iid iid.cpp)
tempi_test_exe(numeric numeric.cpp)
tempi_test_exe(measure_system measure_system.cpp)
tempi_test_exe(type_equivalence type_equivalence.cpp)
tempi_test_exe(init_finalize init_finalize.cpp)
tempi_test_exe(type_commit type_commit.cpp)
tempi_test_exe(pack_unpack pack_unpack.cpp)
tempi_test_exe(isend isend.cu)
tempi_test_exe(isend_contiguous isend_contiguous.cu)
tempi_test_exe(sender sender.cpp)
tempi_test_exe(send send.cpp)
tempi_test_exe(send_vector send_vector.cpp)
tempi_test_exe(dist_graph_create_adjacent dist_graph_create_adjacent.cpp)

if(TEMPI_ENABLE_KAHIP)
  tempi_test_exe(partition_kahip partition_kahip.cpp)

  # KaHIP process mapping often returns unbalanced partitions
  tempi_test_exe(partition_kahip_process_mapping partition_kahip_process_mapping.cpp)
endif()
