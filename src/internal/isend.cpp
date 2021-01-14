//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "isend.hpp"

#include "async_operation.hpp"
#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "topology.hpp"
#include "type_cache.hpp"

int isend::impl(PARAMS_MPI_Isend) {
  LOG_SPEW("isend::impl");

  dest = topology::library_rank(comm, dest);

  // use library MPI for memory we can't reach on the device
  cudaPointerAttributes attr = {};
  CUDA_RUNTIME(cudaPointerGetAttributes(&attr, buf));
  if (nullptr == attr.devicePointer) {
    LOG_SPEW("isend::impl: use library (host memory)");
    int err = libmpi.MPI_Isend(ARGS_MPI_Isend);
    async::try_progress();
    return err;
  }

  // if the type has a packer, use the managed request
  auto pi = typeCache.find(datatype);
  if (typeCache.end() != pi && pi->second.packer) {
    Packer &packer = *(pi->second.packer);
    *request = *async::start_isend(packer, ARGS_MPI_Isend);
    async::try_progress();
    return MPI_SUCCESS;
  }

  // if all else fails, just do MPI_Send
  LOG_SPEW("isend::impl: use library (fallthrough)");
  int err = libmpi.MPI_Isend(ARGS_MPI_Isend);
  async::try_progress();
  return err;
}