##          Copyright Carl Pearson 2020 - 2021.
## Distributed under the Boost Software License, Version 1.0.
##    (See accompanying file LICENSE or copy at
##    https://www.boost.org/LICENSE_1_0.txt)

set(TEMPI_SOURCES ${TEMPI_SOURCES}
${CMAKE_CURRENT_LIST_DIR}/alltoallv.cpp
${CMAKE_CURRENT_LIST_DIR}/comm_rank.cpp
${CMAKE_CURRENT_LIST_DIR}/dist_graph_create_adjacent.cpp
${CMAKE_CURRENT_LIST_DIR}/dist_graph_neighbors.cpp
${CMAKE_CURRENT_LIST_DIR}/finalize.cpp
${CMAKE_CURRENT_LIST_DIR}/init_thread.cpp
${CMAKE_CURRENT_LIST_DIR}/init.cpp
${CMAKE_CURRENT_LIST_DIR}/irecv.cpp
${CMAKE_CURRENT_LIST_DIR}/isend.cpp
${CMAKE_CURRENT_LIST_DIR}/neighbor_alltoallv.cpp
${CMAKE_CURRENT_LIST_DIR}/neighbor_alltoallw.cpp
${CMAKE_CURRENT_LIST_DIR}/pack.cpp
${CMAKE_CURRENT_LIST_DIR}/recv.cpp
${CMAKE_CURRENT_LIST_DIR}/send.cpp
${CMAKE_CURRENT_LIST_DIR}/type_commit.cpp
${CMAKE_CURRENT_LIST_DIR}/type_free.cpp
${CMAKE_CURRENT_LIST_DIR}/unpack.cpp
${CMAKE_CURRENT_LIST_DIR}/wait.cpp

${CMAKE_CURRENT_LIST_DIR}/internal/alltoallv_impl.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/allocators.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/async_operation.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/benchmark.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/counters.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/env.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/events.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/iid.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/irecv.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/isend.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/measure_system.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/measure_system.cu
${CMAKE_CURRENT_LIST_DIR}/internal/neighbor_alltoallw.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/packer.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/packer_1d.cu
${CMAKE_CURRENT_LIST_DIR}/internal/packer_2d.cu
${CMAKE_CURRENT_LIST_DIR}/internal/packer_3d.cu
${CMAKE_CURRENT_LIST_DIR}/internal/partition.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/request.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/send.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/sender.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/statistics.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/streams.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/symbols.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/tags.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/topology.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/type_cache.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/types.cpp
)

if(TEMPI_ENABLE_METIS)
  set(TEMPI_SOURCES ${TEMPI_SOURCES} ${CMAKE_CURRENT_LIST_DIR}/internal/partition_metis.cpp)
endif()

if(TEMPI_ENABLE_KAHIP)
  set(TEMPI_SOURCES ${TEMPI_SOURCES} ${CMAKE_CURRENT_LIST_DIR}/internal/partition_kahip.cpp)
  set(TEMPI_SOURCES ${TEMPI_SOURCES} ${CMAKE_CURRENT_LIST_DIR}/internal/partition_kahip_process_mapping.cpp)
endif()  

set(TEMPI_SOURCES 
  ${TEMPI_SOURCES}
  PARENT_SCOPE
)