//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "allocators.hpp"
#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "symbols.hpp"
#include "topology.hpp"
#include "type_cache.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <vector>

extern "C" int MPI_Recv(PARAMS_MPI_Recv) {
  if (environment::noTempi) {
    return libmpi.MPI_Recv(ARGS_MPI_Recv);
  }

  source = topology::library_rank(comm, source);

  // use library MPI for memory we can't reach on the device
  if (nullptr == buf) {
    LOG_WARN("called MPI_Recv on buf=nullptr");
    return libmpi.MPI_Recv(ARGS_MPI_Recv);
  } else {
    cudaPointerAttributes attr = {};
    cudaError_t err = cudaPointerGetAttributes(&attr, buf);
    cudaGetLastError(); // clear error
    if (err == cudaErrorInvalidValue || nullptr == attr.devicePointer) {
      LOG_SPEW("MPI_Recv: use library (host memory)");
      return libmpi.MPI_Recv(ARGS_MPI_Recv);
    }
    CUDA_RUNTIME(err);
  }

  auto pi = typeCache.find(datatype);

  // if sender is found
  if (typeCache.end() != pi && pi->second.recver) {
    LOG_SPEW("MPI_Recv: cached Recver");
    return pi->second.recver->recv(ARGS_MPI_Recv);
  }

  // if all else fails, just call MPI_Recv
  LOG_SPEW("MPI_Recv: use library (fallthrough)");
  return libmpi.MPI_Recv(ARGS_MPI_Recv);
}
