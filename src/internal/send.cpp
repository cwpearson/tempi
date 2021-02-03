//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "send.hpp"

#include "allocators.hpp"
#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "measure_system.hpp"
#include "topology.hpp"
#include "type_cache.hpp"

#include <optional>
#if TEMPI_OUTPUT_LEVEL >= 4
#include <sstream>
#endif

int send::impl(PARAMS_MPI_Send) {
  LOG_SPEW("in send::impl");

  dest = topology::library_rank(comm, dest);

  // use library MPI for memory we can't reach on the device
  cudaPointerAttributes attr = {};
  CUDA_RUNTIME(cudaPointerGetAttributes(&attr, buf));
  if (nullptr == attr.devicePointer) {
    LOG_SPEW("send::impl: use library (host memory)");
    return libmpi.MPI_Send(ARGS_MPI_Send);
  }

  auto pi = typeCache.find(datatype);

  // if sender is found
  if (typeCache.end() != pi && pi->second.sender) {
    LOG_SPEW("send::impl: cached Sender");
    assert(pi->second.sender);
    return pi->second.sender->send(ARGS_MPI_Send);
  }

  // if all else fails, just do MPI_Send
  LOG_SPEW("send::impl: use library (fallthrough)");
  return libmpi.MPI_Send(ARGS_MPI_Send);
}