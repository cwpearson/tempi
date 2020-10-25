set(TEMPI_SOURCES ${TEMPI_SOURCES}
${CMAKE_CURRENT_LIST_DIR}/alltoallv.cpp
${CMAKE_CURRENT_LIST_DIR}/dist_graph_create_adjacent.cpp
${CMAKE_CURRENT_LIST_DIR}/finalize.cpp
${CMAKE_CURRENT_LIST_DIR}/init.cpp
${CMAKE_CURRENT_LIST_DIR}/isend.cpp
${CMAKE_CURRENT_LIST_DIR}/isend.cpp
${CMAKE_CURRENT_LIST_DIR}/neighbor_alltoallv.cpp
${CMAKE_CURRENT_LIST_DIR}/pack.cpp
${CMAKE_CURRENT_LIST_DIR}/recv.cpp
${CMAKE_CURRENT_LIST_DIR}/send.cpp
${CMAKE_CURRENT_LIST_DIR}/type_commit.cpp
${CMAKE_CURRENT_LIST_DIR}/wait.cpp

${CMAKE_CURRENT_LIST_DIR}/internal/allocators.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/env.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/packer_stride_1.cu
${CMAKE_CURRENT_LIST_DIR}/internal/packer_stride_2.cu
${CMAKE_CURRENT_LIST_DIR}/internal/requests.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/streams.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/symbols.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/task_staged_isend.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/topology.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/types.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/worker.cpp
)

set(TEMPI_SOURCES 
  ${TEMPI_SOURCES}
  PARENT_SCOPE
)