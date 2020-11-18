#include "packer.hpp"

#include "cuda_runtime.hpp"
#include "logging.hpp"
#include "streams.hpp"

Packer::LaunchInfo Packer::pack_launch_info(const void *inbuf) {
  cudaPointerAttributes attr;
  CUDA_RUNTIME(cudaPointerGetAttributes(&attr, inbuf));
  Packer::LaunchInfo ret{.stream = nullptr, .device = -1};

  if (0 >= attr.device) {
    ret.stream = kernStream[attr.device];
    ret.device = attr.device;
  }
  return ret;
}

Packer::LaunchInfo Packer::unpack_launch_info(const void *outbuf) {
  cudaPointerAttributes attr;
  CUDA_RUNTIME(cudaPointerGetAttributes(&attr, outbuf));
  Packer::LaunchInfo ret{.stream = nullptr, .device = -1};

  if (0 >= attr.device) {
    ret.stream = kernStream[attr.device];
    ret.device = attr.device;
  }
  return ret;
}

#if 0
void Packer::sync(const void *inbuf) {
  Packer::LaunchInfo info = pack_launch_info(inbuf);
  if (!info.stream) {
    LOG_ERROR("requested Packer sync for "
              << uintptr_t(inbuf)
              << " which does not have an associated stream");
  } else {
    CUDA_RUNTIME(cudaStreamSynchronize(info.stream));
  }
}
#endif