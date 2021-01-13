//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "packer_1d.hpp"

#include "cuda_runtime.hpp"
#include "dim3.hpp"
#include "logging.hpp"

Packer1D::Packer1D(unsigned off, unsigned extent)
    : offset_(off), extent_(extent) {}

void Packer1D::launch_pack(void *outbuf, int *position, const void *inbuf,
                           const int incount, cudaStream_t stream) const {
  assert(position);
  assert(outbuf);
  assert(inbuf);
  inbuf = static_cast<const char *>(inbuf) + offset_;
  outbuf = static_cast<char *>(outbuf) + *position;
  const uint64_t nBytes = incount * extent_;
  // cudaMemcpy is not synchronous for d2d
  CUDA_RUNTIME(
      cudaMemcpyAsync(outbuf, inbuf, nBytes, cudaMemcpyDefault, stream));
  CUDA_RUNTIME(cudaStreamSynchronize(stream));
  (*position) += incount * extent_;
}

void Packer1D::launch_unpack(const void *inbuf, int *position, void *outbuf,
                             const int outcount, cudaStream_t stream) const {
  assert(position);
  assert(outbuf);
  assert(inbuf);
  outbuf = static_cast<char *>(outbuf) + offset_;
  inbuf = static_cast<const char *>(inbuf) + *position;
  const uint64_t nBytes = outcount * extent_;
  // cudaMemcpy is not synchronous for d2d
  CUDA_RUNTIME(
      cudaMemcpyAsync(outbuf, inbuf, nBytes, cudaMemcpyDefault, stream));
  CUDA_RUNTIME(cudaStreamSynchronize(stream));
  (*position) += outcount * extent_;
}

void Packer1D::pack_async(void *outbuf, int *position, const void *inbuf,
                          const int incount) const {
  LaunchInfo info = pack_launch_info(inbuf);
  launch_pack(outbuf, position, inbuf, incount, info.stream);
}

// same as async but synchronize after launch
void Packer1D::pack(void *outbuf, int *position, const void *inbuf,
                    const int incount) const {
  LaunchInfo info = pack_launch_info(inbuf);
  launch_pack(outbuf, position, inbuf, incount, info.stream);
  CUDA_RUNTIME(cudaStreamSynchronize(info.stream));
}

void Packer1D::unpack(const void *inbuf, int *position, void *outbuf,
                      const int outcount) const {
  LaunchInfo info = unpack_launch_info(outbuf);
  launch_unpack(inbuf, position, outbuf, outcount, info.stream);
}