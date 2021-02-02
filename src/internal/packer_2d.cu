//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "pack_kernels.cuh"
#include "packer_2d.hpp"

#include "counters.hpp"
#include "cuda_runtime.hpp"
#include "dim3.hpp"
#include "logging.hpp"

Packer2D::Packer2D(unsigned off, unsigned blockLength, unsigned count,
                   unsigned stride, unsigned extent)
    : offset_(off), blockLength_(blockLength), count_(count), stride_(stride),
      extent_(extent), config_(off, blockLength, count) {
  assert(blockLength_ > 0);

}

void Packer2D::launch_pack(void *outbuf, int *position, const void *inbuf,
                           const int incount, cudaStream_t stream,
                           cudaEvent_t kernelStart,
                           cudaEvent_t kernelStop) const {
  TEMPI_COUNTER_OP(pack2d, NUM_PACKS, ++);
  inbuf = static_cast<const char *>(inbuf) + offset_;

  const dim3 gd = config_.dim_grid(incount);
  const dim3 bd = config_.dim_block();
  if (kernelStart) {
    CUDA_RUNTIME(cudaEventRecord(kernelStart, stream));
  }
  outbuf = static_cast<char *>(outbuf) + *position;
  LOG_SPEW("packfn_");
  config_.packfn<<<gd, bd, 0, stream>>>(outbuf, inbuf, incount, blockLength_, count_,
                                  stride_, extent_);
  if (kernelStop) {
    CUDA_RUNTIME(cudaEventRecord(kernelStop, stream));
  }
  CUDA_RUNTIME(cudaGetLastError());
  (*position) += incount * count_ * blockLength_;
}

void Packer2D::launch_unpack(const void *inbuf, int *position, void *outbuf,
                             const int outcount, cudaStream_t stream,
                             cudaEvent_t kernelStart,
                             cudaEvent_t kernelStop) const {
  TEMPI_COUNTER_OP(pack2d, NUM_UNPACKS, ++);
  outbuf = static_cast<char *>(outbuf) + offset_;

  const dim3 gd = config_.dim_grid(outcount);
  const dim3 bd = config_.dim_block();

  if (kernelStart) {
    CUDA_RUNTIME(cudaEventRecord(kernelStart, stream));
  }

  outbuf = static_cast<char *>(outbuf) + *position;
  config_.unpackfn<<<gd, bd, 0, stream>>>(outbuf, inbuf, outcount, blockLength_,
                                    count_, stride_, extent_);

  if (kernelStop) {
    CUDA_RUNTIME(cudaEventRecord(kernelStop, stream));
  }
  CUDA_RUNTIME(cudaGetLastError());
  (*position) += outcount * count_ * blockLength_;
}

void Packer2D::pack_async(void *outbuf, int *position, const void *inbuf,
                          const int incount, cudaEvent_t event) const {
  LaunchInfo info = pack_launch_info(inbuf);
  launch_pack(outbuf, position, inbuf, incount, info.stream);
  if (event) {
    CUDA_RUNTIME(cudaEventRecord(event, info.stream));
  }
}

void Packer2D::unpack_async(const void *inbuf, int *position, void *outbuf,
                            const int outcount, cudaEvent_t event) const {
  LaunchInfo info = unpack_launch_info(outbuf);
  launch_unpack(inbuf, position, outbuf, outcount, info.stream);
  if (event) {
    CUDA_RUNTIME(cudaEventRecord(event, info.stream));
  }
}

// same as async but synchronize after launch
void Packer2D::pack(void *outbuf, int *position, const void *inbuf,
                    const int incount) const {
  LaunchInfo info = pack_launch_info(inbuf);
  launch_pack(outbuf, position, inbuf, incount, info.stream);
  CUDA_RUNTIME(cudaStreamSynchronize(info.stream));
}

void Packer2D::unpack(const void *inbuf, int *position, void *outbuf,
                      const int outcount) const {
  LaunchInfo info = unpack_launch_info(outbuf);
  launch_unpack(inbuf, position, outbuf, outcount, info.stream);
  CUDA_RUNTIME(cudaStreamSynchronize(info.stream));
}