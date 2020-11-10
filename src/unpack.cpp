/* Unlike MPI_Pack, the output buffer needs to be usable after MPI_Unpack.
*/

#include "logging.hpp"

#include "cuda_runtime.hpp"
#include "env.hpp"
#include "symbols.hpp"
#include "types.hpp"

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvToolsExt.h>

extern "C" int MPI_Unpack(PARAMS_MPI_Unpack) {

  // find the underlying MPI call
  static Func_MPI_Unpack fn = libmpi.MPI_Unpack;
  assert(fn);

  if (environment::noTempi) {
    return fn(ARGS_MPI_Unpack);
  }

  nvtxRangePush("MPI_Unpack");
  int err = MPI_ERR_UNKNOWN;

  bool enabled = true;
  enabled &= !environment::noPack;
  if (!enabled) {
    LOG_SPEW("system MPI_Unpack: disabled by env");
    err = fn(ARGS_MPI_Unpack);
    goto cleanup_and_exit;
  }

  if (packerCache.count(datatype)) {
    // only optimize device-to-device unpack
    cudaPointerAttributes outAttrs = {}, inAttrs = {};
    CUDA_RUNTIME(cudaPointerGetAttributes(&outAttrs, outbuf));
    CUDA_RUNTIME(cudaPointerGetAttributes(&inAttrs, inbuf));

    // if the data can be accessed from the GPU, use the GPU
    bool isOutDev = outAttrs.devicePointer;
    bool isInDev = inAttrs.devicePointer;

    if (!isOutDev || !isInDev) {
      LOG_SPEW("system MPI_Unpack: not device-device");
      err = fn(ARGS_MPI_Unpack);
      goto cleanup_and_exit;
    }
    std::shared_ptr<Packer> packer = packerCache[datatype];
    CUDA_RUNTIME(cudaSetDevice(inAttrs.device));
    packer->unpack(inbuf, position, outbuf, outcount);
    err = MPI_SUCCESS;
    goto cleanup_and_exit;
  } else {
    LOG_SPEW("system MPI_Unpack: no packer");
    err = fn(ARGS_MPI_Unpack);
    goto cleanup_and_exit;
  }

cleanup_and_exit:
  nvtxRangePop();
  return err;
}