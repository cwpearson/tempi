#include "streams.hpp"

#include "cuda_runtime.hpp"
#include "logging.hpp"

#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h> // nvtxNameCudaStreamA

#include <string>

/*extern*/ cudaStream_t commStream = {};
/*extern*/ cudaStream_t kernStream = {};

void streams_init() {
  nvtxRangePush("streams_init");
  int count;
  CUDA_RUNTIME(cudaGetDeviceCount(&count));

  CUDA_RUNTIME(cudaStreamCreate(&commStream));
  nvtxNameCudaStreamA(commStream, "TEMPI_comm");
  CUDA_RUNTIME(cudaStreamCreate(&kernStream));
  nvtxNameCudaStreamA(kernStream, "TEMPI_kern");
  nvtxRangePop();
}
void streams_finalize() {
  nvtxRangePush("streams_finalize");
  CUDA_RUNTIME(cudaStreamSynchronize(commStream));
  CUDA_RUNTIME(cudaStreamDestroy(commStream));
  CUDA_RUNTIME(cudaStreamSynchronize(kernStream));
  CUDA_RUNTIME(cudaStreamDestroy(kernStream));
  kernStream = {};
  commStream = {};
  nvtxRangePop();
}
