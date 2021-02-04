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

  // use library MPI for memory we can't reach on the device
  cudaPointerAttributes attr = {};
  CUDA_RUNTIME(cudaPointerGetAttributes(&attr, buf));
  if (nullptr == attr.devicePointer) {
    LOG_SPEW("irecv::impl: use library (host memory)");
    int err = libmpi.MPI_Irecv(ARGS_MPI_Irecv);
    async::try_progress();
    return err;
  }

  // use fast path for non-contigous data where a packer exists
  auto pi = typeCache.find(datatype);
  if (typeCache.end() != pi && pi->second.desc.ndims() > 1 && pi->second.packer) {
    Packer &packer = *(pi->second.packer);
    const StridedBlock &sb = pi->second.desc;
    async::start_irecv(sb, packer, ARGS_MPI_Irecv);
    async::try_progress();
    return MPI_SUCCESS;
  }

  LOG_SPEW("irecv::impl: use library (fallthrough)");
  int err = libmpi.MPI_Irecv(ARGS_MPI_Irecv);
  async::try_progress();
  return err;
}