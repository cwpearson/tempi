/*
 The implementation defines what the output of MPI_Pack is: it is allowed to
 prefix or postfix packed data with additional information. And future calls to
 MPI_Send should use the same communicator, and MPI_Packed datatype

 So, this timplementation makes the packing asynchronous, and before calls with
 a packable datatype in communication routines, we will synchronize there. This
 removes the synch overhead at each call
*/

#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "symbols.hpp"
#include "types.hpp"

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvToolsExt.h>

extern "C" int MPI_Pack(PARAMS_MPI_Pack) {
  if (environment::noTempi) {
    return libmpi.MPI_Pack(ARGS_MPI_Pack);
  }
  if (environment::noPack) {
    LOG_SPEW("library MPI_Pack: disabled by env");
    return libmpi.MPI_Pack(ARGS_MPI_Pack);
  }

  nvtxRangePush("MPI_Pack");
  int err = MPI_ERR_UNKNOWN;

  auto pi = packerCache.find(datatype);
  if (packerCache.end() != pi) {
    // only optimize device-to-device pack
    cudaPointerAttributes outAttrs{}, inAttrs{};
    CUDA_RUNTIME(cudaPointerGetAttributes(&outAttrs, outbuf));
    CUDA_RUNTIME(cudaPointerGetAttributes(&inAttrs, inbuf));

    // if the data can be accessed from the GPU, use the GPU
    bool isOutDev = outAttrs.devicePointer;
    bool isInDev = inAttrs.devicePointer;

    if (!isOutDev || !isInDev) {
      LOG_SPEW("library MPI_Pack: not device-device");
      err = libmpi.MPI_Pack(ARGS_MPI_Pack);
      goto cleanup_and_exit;
    }
    pi->second->pack(outbuf, position, inbuf, incount);
    err = MPI_SUCCESS;
    goto cleanup_and_exit;
  } else {
    LOG_SPEW("library MPI_Pack: no packer");
    err = libmpi.MPI_Pack(ARGS_MPI_Pack);
    goto cleanup_and_exit;
  }

cleanup_and_exit:
  nvtxRangePop();
  return err;
}