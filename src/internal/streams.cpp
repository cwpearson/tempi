#include "streams.hpp"

#include "cuda_runtime.hpp"
#include "logging.hpp"

#include <nvToolsExt.h>
#include <nvToolsExtCudaRt.h> // nvtxNameCudaStreamA

#include <string>

/*extern*/ std::vector<cudaStream_t> commStream = {};
/*extern*/ std::vector<cudaStream_t> kernStream = {};

void streams_init() {
  nvtxRangePush("streams_init");
  int count;
  CUDA_RUNTIME(cudaGetDeviceCount(&count));

  LOG_SPEW("create " << count << " streams");
  commStream = std::vector<cudaStream_t>(count, {});
  kernStream = std::vector<cudaStream_t>(count, {});
  for (int i = 0; i < count; ++i) {
    CUDA_RUNTIME(cudaSetDevice(i));
    CUDA_RUNTIME(cudaStreamCreate(&commStream[i]));
    nvtxNameCudaStreamA(commStream[i],
                        ("campi_comm_" + std::to_string(i)).c_str());
    CUDA_RUNTIME(cudaStreamCreate(&kernStream[i]));
    nvtxNameCudaStreamA(kernStream[i],
                        ("campi_kern_" + std::to_string(i)).c_str());
  }
  nvtxRangePop();
}
void streams_finalize() {
  nvtxRangePush("streams_finalize");
  for (size_t i = 0; i < commStream.size(); ++i) {
    CUDA_RUNTIME(cudaStreamSynchronize(commStream[i]));
    CUDA_RUNTIME(cudaStreamDestroy(commStream[i]));
    CUDA_RUNTIME(cudaStreamSynchronize(kernStream[i]));
    CUDA_RUNTIME(cudaStreamDestroy(kernStream[i]));
  }
  commStream.clear();
  kernStream.clear();
  nvtxRangePop();
}
