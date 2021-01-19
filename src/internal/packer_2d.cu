//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "pack_kernels_2d.cuh"
#include "packer_2d.hpp"

#include "counters.hpp"
#include "cuda_runtime.hpp"
#include "dim3.hpp"
#include "logging.hpp"

Packer2D::Packer2D(unsigned off, unsigned blockLength, unsigned count,
                   unsigned stride)
    : params_(blockLength, count) {
  offset_ = off;
  blockLength_ = blockLength;
  assert(blockLength_ > 0);
  count_ = count;
  stride_ = stride;

  // blocklength is a multiple of wordsize
  // offset is a multiple of wordsize
  // wordsize is at most 8
  wordSize_ = 1;
  while (0 == blockLength % (wordSize_ * 2) && 0 == offset_ % (wordSize_ * 2) &&
         (wordSize_ * 2 <= 8)) {
    wordSize_ *= 2;
  }

  // griddim.z should be incount
  bd_ = Dim3::fill_xyz_by_pow2(Dim3(blockLength_ / wordSize_, count_, 1), 512);
  gd_ = (Dim3(blockLength_ / wordSize_, count_, 1) + bd_ - Dim3(1, 1, 1)) / bd_;
}

void Packer2D::launch_pack(void *outbuf, int *position, const void *inbuf,
                           const int incount, cudaStream_t stream,
                           cudaEvent_t kernelStart,
                           cudaEvent_t kernelStop) const {
  TEMPI_COUNTER_OP(pack2d, NUM_PACKS, ++);
  inbuf = static_cast<const char *>(inbuf) + offset_;

#ifdef USE_NEW_PACKER
  dim3 gd = params_.dim_grid(incount);
  dim3 bd = params_.dim_block();
#else
  Dim3 gd = gd_;
  gd.z = incount;
#endif
  if (kernelStart) {
    CUDA_RUNTIME(cudaEventRecord(kernelStart, stream));
  }
#ifdef USE_NEW_PACKER
  outbuf = static_cast<char *>(outbuf) + *position;
  params_.packfn<<<gd, bd, 0, stream>>>(outbuf, inbuf, incount, blockLength_,
                                        count_, stride_);
#else
  // LOG_SPEW("wordSize_ = " << wordSize_);
  if (uintptr_t(inbuf) % wordSize_) {
    LOG_WARN("pack kernel may be unaligned.");
  }
  if (4 == wordSize_) {
    pack_bytes<4><<<gd, bd_, 0, stream>>>(outbuf, *position, inbuf, incount,
                                          blockLength_, count_, stride_);
  } else if (8 == wordSize_) {
    pack_bytes<8><<<gd, bd_, 0, stream>>>(outbuf, *position, inbuf, incount,
                                          blockLength_, count_, stride_);
  } else if (2 == wordSize_) {
    pack_bytes<2><<<gd, bd_, 0, stream>>>(outbuf, *position, inbuf, incount,
                                          blockLength_, count_, stride_);
  } else {
    pack_bytes<1><<<gd, bd_, 0, stream>>>(outbuf, *position, inbuf, incount,
                                          blockLength_, count_, stride_);
  }
#endif

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

  Dim3 gd = gd_;
  gd.z = outcount;
  // LOG_SPEW("wordSize_ = " << wordSize_);
  if (kernelStart) {
    CUDA_RUNTIME(cudaEventRecord(kernelStart, stream));
  }
  if (4 == wordSize_) {
    unpack_bytes<4><<<gd, bd_, 0, stream>>>(outbuf, *position, inbuf, outcount,
                                            blockLength_, count_, stride_);
  } else if (8 == wordSize_) {
    unpack_bytes<8><<<gd, bd_, 0, stream>>>(outbuf, *position, inbuf, outcount,
                                            blockLength_, count_, stride_);
  } else if (2 == wordSize_) {
    unpack_bytes<2><<<gd, bd_, 0, stream>>>(outbuf, *position, inbuf, outcount,
                                            blockLength_, count_, stride_);
  } else {
    unpack_bytes<1><<<gd, bd_, 0, stream>>>(outbuf, *position, inbuf, outcount,
                                            blockLength_, count_, stride_);
  }
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