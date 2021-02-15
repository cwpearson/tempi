//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

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
  CUDA_RUNTIME(cudaStreamCreateWithFlags(&commStream, cudaStreamNonBlocking));
  nvtxNameCudaStreamA(commStream, "TEMPI_comm");
  CUDA_RUNTIME(cudaStreamCreateWithFlags(&kernStream, cudaStreamNonBlocking));
  nvtxNameCudaStreamA(kernStream, "TEMPI_kern");
  nvtxRangePop();
}
void streams_finalize() {
  nvtxRangePush("streams_finalize");
  if (commStream) {
    CUDA_RUNTIME(cudaStreamSynchronize(commStream));
    CUDA_RUNTIME(cudaStreamDestroy(commStream));
    commStream = {};
  }
  if (kernStream) {
    CUDA_RUNTIME(cudaStreamSynchronize(kernStream));
    CUDA_RUNTIME(cudaStreamDestroy(kernStream));
    kernStream = {};
  }
  nvtxRangePop();
}
