//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

/* Unlike MPI_Pack, the output buffer needs to be usable after MPI_Unpack.
 */

#include "logging.hpp"

#include "cuda_runtime.hpp"
#include "env.hpp"
#include "type_cache.hpp"
#include "symbols.hpp"

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvToolsExt.h>

extern "C" int MPI_Unpack(PARAMS_MPI_Unpack) {
  if (environment::noTempi) {
    return libmpi.MPI_Unpack(ARGS_MPI_Unpack);
  }
  if (environment::noPack) {
    LOG_SPEW("system MPI_Unpack: disabled by env");
    return libmpi.MPI_Unpack(ARGS_MPI_Unpack);
  }

  nvtxRangePush("MPI_Unpack");
  int err = MPI_ERR_UNKNOWN;

  auto pi = typeCache.find(datatype);
  if (typeCache.end() != pi) {
    // only optimize device-to-device unpack
    cudaPointerAttributes outAttrs = {}, inAttrs = {};
    CUDA_RUNTIME(cudaPointerGetAttributes(&outAttrs, outbuf));
    CUDA_RUNTIME(cudaPointerGetAttributes(&inAttrs, inbuf));

    // if the data can be accessed from the GPU, use the GPU
    bool isOutDev = outAttrs.devicePointer;
    bool isInDev = inAttrs.devicePointer;

    if (!isOutDev || !isInDev) {
      LOG_SPEW("system MPI_Unpack: not device-device");
      err = libmpi.MPI_Unpack(ARGS_MPI_Unpack);
      goto cleanup_and_exit;
    }
    pi->second.packer->unpack(inbuf, position, outbuf, outcount);
    err = MPI_SUCCESS;
    goto cleanup_and_exit;
  } else {
    LOG_SPEW("system MPI_Unpack: no packer");
    err = libmpi.MPI_Unpack(ARGS_MPI_Unpack);
    goto cleanup_and_exit;
  }

cleanup_and_exit:
  nvtxRangePop();
  return err;
}