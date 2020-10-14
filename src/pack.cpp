#include "logging.hpp"

#include "cuda_runtime.hpp"
#include "env.hpp"
#include "types.hpp"

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvToolsExt.h>

#include <dlfcn.h>

// #include <vector>

#define PARAMS                                                                 \
  const void *inbuf, int incount, MPI_Datatype datatype, void *outbuf,         \
      int outsize, int *position, MPI_Comm comm

#define ARGS inbuf, incount, datatype, outbuf, outsize, position, comm

extern "C" int MPI_Pack(PARAMS) {

  // find the underlying MPI call
  typedef int (*Func_MPI_Pack)(PARAMS);
  static Func_MPI_Pack fn = nullptr;
  if (!fn) {
    fn = reinterpret_cast<Func_MPI_Pack>(dlsym(RTLD_NEXT, "MPI_Pack"));
  }
  TEMPI_DISABLE_GUARD;
  nvtxRangePush("MPI_Pack");
  LOG_DEBUG("MPI_Pack");
  int err = MPI_ERR_UNKNOWN;

  bool enabled = true;
  enabled &= !environment::noPack;
  if (!enabled) {
    LOG_DEBUG("library MPI_Pack: disabled by env");
    err = fn(ARGS);
    goto cleanup_and_exit;
  }

  if (packerCache.count(datatype)) {
    // only optimize device-to-device pack
    cudaPointerAttributes outAttrs = {}, inAttrs = {};
    CUDA_RUNTIME(cudaPointerGetAttributes(&outAttrs, outbuf));
    CUDA_RUNTIME(cudaPointerGetAttributes(&inAttrs, inbuf));

    // if the data can be accessed from the GPU, use the GPU
    bool outDev = outAttrs.devicePointer;
    bool inDev = inAttrs.devicePointer;

    if (!outDev || !inDev) {
      LOG_DEBUG("library MPI_Pack: not device-device");
      err = fn(ARGS);
      goto cleanup_and_exit;
    }
    std::shared_ptr<Packer> packer = packerCache[datatype];
    CUDA_RUNTIME(cudaSetDevice(inAttrs.device));
    packer->pack(outbuf, position, inbuf, incount);
    err = MPI_SUCCESS;
    goto cleanup_and_exit;
  } else {
    LOG_DEBUG("library MPI_Pack");
    err = fn(ARGS);
    goto cleanup_and_exit;
  }

cleanup_and_exit:
  nvtxRangePop();
  return err;
}