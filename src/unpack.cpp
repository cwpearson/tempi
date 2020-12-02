/* Unlike MPI_Pack, the output buffer needs to be usable after MPI_Unpack.
 */

#include "logging.hpp"

#include "cuda_runtime.hpp"
#include "env.hpp"
#include "packer_cache.hpp"
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

  auto pi = packerCache.find(datatype);
  if (packerCache.end() != pi) {
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
    pi->second->unpack(inbuf, position, outbuf, outcount);
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