#include "logging.hpp"

#include "cuda_runtime.hpp"
#include "types.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <dlfcn.h>

#include <cstdlib>
#include <vector>

#define PARAMS                                                                 \
  const void *inbuf, int incount, MPI_Datatype datatype, void *outbuf,         \
      int outsize, int *position, MPI_Comm comm

#define ARGS inbuf, incount, datatype, outbuf, outsize, position, comm

extern "C" int MPI_Pack(PARAMS) {
  LOG_DEBUG("MPI_Pack");

  // find the underlying MPI call
  typedef int (*Func_MPI_Pack)(PARAMS);
  static Func_MPI_Pack fn = nullptr;
  if (!fn) {
    fn = reinterpret_cast<Func_MPI_Pack>(dlsym(RTLD_NEXT, "MPI_Pack"));
  }

  bool enabled = true;
  enabled &= (nullptr == std::getenv("SCAMPI_NO_PACK"));
  if (!enabled) {
    LOG_DEBUG("library MPI_Pack: disabled by env");
    return fn(ARGS);
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
      return fn(ARGS);
    }
    std::shared_ptr<Packer> packer = packerCache[datatype];
    CUDA_RUNTIME(cudaSetDevice(inAttrs.device));
    packer->pack(outbuf, position, inbuf, incount);
    return MPI_SUCCESS;
  } else {
    LOG_DEBUG("library MPI_Pack");
    return fn(ARGS);
  }
}