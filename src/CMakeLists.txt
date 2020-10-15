set(TEMPI_SOURCES ${TEMPI_SOURCES}
${CMAKE_CURRENT_LIST_DIR}/alltoallv.cpp
${CMAKE_CURRENT_LIST_DIR}/finalize.cpp
${CMAKE_CURRENT_LIST_DIR}/init.cpp
${CMAKE_CURRENT_LIST_DIR}/isend.cpp
${CMAKE_CURRENT_LIST_DIR}/pack.cpp
${CMAKE_CURRENT_LIST_DIR}/recv.cpp
${CMAKE_CURRENT_LIST_DIR}/send.cpp
${CMAKE_CURRENT_LIST_DIR}/type_commit.cpp

${CMAKE_CURRENT_LIST_DIR}/internal/allocator_slab.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/env.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/packer_stride_1.cu
${CMAKE_CURRENT_LIST_DIR}/internal/packer_stride_2.cu
${CMAKE_CURRENT_LIST_DIR}/internal/types.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/streams.cpp
${CMAKE_CURRENT_LIST_DIR}/internal/worker.cpp
)

set(TEMPI_SOURCES 
  ${TEMPI_SOURCES}
  PARENT_SCOPE
)