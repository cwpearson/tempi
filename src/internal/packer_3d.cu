//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

/* TODO: alignment is not really handled in this case.
   we expect all access to be aligned
*/

#include "pack_kernels.cuh"
#include "packer_3d.hpp"

#include "counters.hpp"
#include "cuda_runtime.hpp"
#include "dim3.hpp"
#include "logging.hpp"
#include "streams.hpp"

Packer3D::Packer3D(unsigned off, unsigned blockLength, unsigned count1,
                   unsigned stride1, unsigned count2, unsigned stride2,
                   unsigned extent)
    : offset_(off), blockLength_(blockLength), extent_(extent),
      config_(blockLength, count1, count2) {
  assert(blockLength_ > 0);
  count_[0] = count1;
  count_[1] = count2;
  stride_[0] = stride1;
  stride_[1] = stride2;

#ifndef USE_NEW_PACKER
  // blocklength is a multiple of wordsize
  // offset is a multiple of wordsize
  // wordsize is at most 8
  wordSize_ = 1;
  while (0 == blockLength % (wordSize_ * 2) && 0 == offset_ % (wordSize_ * 2) &&
         (wordSize_ * 2 <= 8)) {
    wordSize_ *= 2;
  }

  bd_ = Dim3::fill_xyz_by_pow2(
      Dim3(blockLength_ / wordSize_, count_[0], count_[1]), 512);
  gd_ = (Dim3(blockLength_ / wordSize_, count_[0], count_[1]) + bd_ -
         Dim3(1, 1, 1)) /
        bd_;
#endif
}

void Packer3D::launch_pack(void *outbuf, int *position, const void *inbuf,
                           const int incount, cudaStream_t stream) const {
  TEMPI_COUNTER_OP(pack3d, NUM_PACKS, ++);
  LOG_SPEW("launch_pack offset=" << offset_);

  inbuf = static_cast<const char *>(inbuf) + offset_;

#ifdef USE_NEW_PACKER
  outbuf = static_cast<char *>(outbuf) + *position;
  config_.packfn<<<config_.dim_grid(), config_.dim_block(), 0, stream>>>(
      outbuf, inbuf, incount, blockLength_, count_[0], stride_[0], count_[1],
      stride_[1], extent_);
#else
  if (uintptr_t(inbuf) % wordSize_) {
    LOG_WARN("pack kernel may be unaligned.");
  }
  if (4 == wordSize_) {
    LOG_SPEW("wordSize_ = 4");

    pack_3d<4><<<gd_, bd_, 0, stream>>>(outbuf, *position, inbuf, incount,
                                        blockLength_, count_[0], stride_[0],
                                        count_[1], stride_[1]);
  } else if (8 == wordSize_) {
    LOG_SPEW("wordSize_ = 8");
    pack_3d<8><<<gd_, bd_, 0, stream>>>(outbuf, *position, inbuf, incount,
                                        blockLength_, count_[0], stride_[0],
                                        count_[1], stride_[1]);
  } else {
    LOG_SPEW("wordSize == 1");
    pack_3d<1><<<gd_, bd_, 0, stream>>>(outbuf, *position, inbuf, incount,
                                        blockLength_, count_[0], stride_[0],
                                        count_[1], stride_[1]);
  }
#endif
  CUDA_RUNTIME(cudaGetLastError());
  assert(position);
  (*position) += incount * count_[1] * count_[0] * blockLength_;
}

void Packer3D::launch_unpack(const void *inbuf, int *position, void *outbuf,
                             const int outcount, cudaStream_t stream) const {
  TEMPI_COUNTER_OP(pack3d, NUM_UNPACKS, ++);
  outbuf = static_cast<char *>(outbuf) + offset_;

#ifdef USE_NEW_PACKER
  outbuf = static_cast<char *>(outbuf) + *position;
  config_.unpackfn<<<config_.dim_grid(), config_.dim_block(), 0, stream>>>(
      outbuf, inbuf, outcount, blockLength_, count_[0], stride_[0], count_[1],
      stride_[1], extent_);
#else
  if (4 == wordSize_) {
    LOG_SPEW("wordSize_ = 4");
    unpack_3d<4><<<gd_, bd_, 0, stream>>>(outbuf, *position, inbuf, outcount,
                                          blockLength_, count_[0], stride_[0],
                                          count_[1], stride_[1]);

  } else if (8 == wordSize_) {
    LOG_SPEW("wordSize_ = 8");
    unpack_3d<8><<<gd_, bd_, 0, stream>>>(outbuf, *position, inbuf, outcount,
                                          blockLength_, count_[0], stride_[0],
                                          count_[1], stride_[1]);

  } else {
    LOG_SPEW("wordSize == 1");
    unpack_3d<1><<<gd_, bd_, 0, stream>>>(outbuf, *position, inbuf, outcount,
                                          blockLength_, count_[0], stride_[0],
                                          count_[1], stride_[1]);
  }
#endif

  CUDA_RUNTIME(cudaGetLastError());
  assert(position);
  (*position) += outcount * count_[1] * count_[0] * blockLength_;
}

#if 0
void Packer3D::pack_async(void *outbuf, int *position, const void *inbuf,
                               const int incount) const {

  int device;
  CUDA_RUNTIME(cudaGetDevice(&device));
  LaunchInfo info = pack_launch_info(inbuf);
  LOG_SPEW("Packer3D::pack on CUDA " << info.device);
  CUDA_RUNTIME(cudaSetDevice(info.device));
  launch_pack(outbuf, position, inbuf, incount, info.stream);

  LOG_SPEW("Packer3D::restore device " << device);
  CUDA_RUNTIME(cudaSetDevice(device));
}
#endif

void Packer3D::pack_async(void *outbuf, int *position, const void *inbuf,
                          const int incount, cudaEvent_t event) const {
  LaunchInfo info = pack_launch_info(inbuf);
  launch_pack(outbuf, position, inbuf, incount, info.stream);
  if (event) {
    CUDA_RUNTIME(cudaEventRecord(event, info.stream));
  }
}

void Packer3D::unpack_async(const void *inbuf, int *position, void *outbuf,
                            const int outcount, cudaEvent_t event) const {
  LaunchInfo info = unpack_launch_info(outbuf);
  launch_unpack(inbuf, position, outbuf, outcount, info.stream);
  if (event) {
    CUDA_RUNTIME(cudaEventRecord(event, info.stream));
  }
}

// same as async but synchronize after launch
void Packer3D::pack(void *outbuf, int *position, const void *inbuf,
                    const int incount) const {
  LaunchInfo info = pack_launch_info(inbuf);
  launch_pack(outbuf, position, inbuf, incount, info.stream);
  CUDA_RUNTIME(cudaStreamSynchronize(info.stream));
}

void Packer3D::unpack(const void *inbuf, int *position, void *outbuf,
                      const int outcount) const {
  LaunchInfo info = unpack_launch_info(outbuf);
  launch_unpack(inbuf, position, outbuf, outcount, info.stream);
  CUDA_RUNTIME(cudaStreamSynchronize(info.stream));
}
