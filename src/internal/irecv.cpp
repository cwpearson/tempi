//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "irecv.hpp"

#include "async_operation.hpp"
#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "topology.hpp"
#include "type_cache.hpp"

int irecv::impl(PARAMS_MPI_Irecv) {
  LOG_SPEW("irecv::impl");

  source = topology::library_rank(comm, source);

  // let other operations try to make progress
  async::try_progress();

  // use library MPI for memory we can't reach on the device
  cudaPointerAttributes attr = {};
  CUDA_RUNTIME(cudaPointerGetAttributes(&attr, buf));
  if (nullptr == attr.devicePointer) {
    LOG_SPEW("irecv::impl: use library (host memory)");
    return libmpi.MPI_Irecv(ARGS_MPI_Irecv);
  }

  // if the type has a packer, create a managed request
  auto pi = typeCache.find(datatype);
  if (typeCache.end() != pi && pi->second.packer) {
    Packer &packer = *(pi->second.packer);
    *request = *async::start_irecv(packer, ARGS_MPI_Irecv);
    return MPI_SUCCESS;
  }

  LOG_SPEW("irecv::impl: use library (fallthrough)");
  return libmpi.MPI_Irecv(ARGS_MPI_Irecv);
}