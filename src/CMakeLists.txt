set(TEMPI_SOURCES ${TEMPI_SOURCES}
${CMAKE_CURRENT_LIST_DIR}/alltoallv.cpp
${CMAKE_CURRENT_LIST_DIR}/comm_rank.cpp
${CMAKE_CURRENT_LIST_DIR}/dist_graph_create_adjacent.cpp
${CMAKE_CURRENT_LIST_DIR}/dist_graph_neighbors.cpp
${CMAKE_CURRENT_LIST_DIR}/finalize.cpp
${CMAKE_CURRENT_LIST_DIR}/init.cpp
${CMAKE_CURRENT_LIST_DIR}/isend.cpp
${CMAKE_CURRENT_LIST_DIR}/isend.cpp
${CMAKE_CURRENT_LIST_DIR}/neighbor_alltoallv.cpp
${CMAKE_CURRENT_LIST_DIR}/pack.cpp
${CMAKE_CURRENT_LIST_DIR}/recv.cpp
${CMAKE_CURRENT_LIST_DIR}/send.cpp
${CMAKE_CURRENT_LIST_DIR}/type_commit.cpp
${CMAKE_CURRENT_LIST_DIR}/type_free.cpp
${CMAKE_CURRENT_LIST_DIR}/unpack.cpp
${CMAKE_CURRENT_LIST_DIR}/wait.cpp

${CMAKE_CURRENT_LIST_DIR}/internal/alltoallv_impl.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/allocators.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/env.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/pack_kernels.cu
${CMAKE_CURRENT_LIST_DIR}/internal/packer.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/packer_1d.cu
${CMAKE_CURRENT_LIST_DIR}/internal/packer_2d.cu
${CMAKE_CURRENT_LIST_DIR}/internal/packer_3d.cu
${CMAKE_CURRENT_LIST_DIR}/internal/packer_cache.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/partition.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/requests.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/streams.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/symbols.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/task_staged_isend.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/topology.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/types.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/worker.cpp
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