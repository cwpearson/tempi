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
      packfn_(nullptr), unpackfn_(nullptr) {
  assert(blockLength_ > 0);
  count_[0] = count1;
  count_[1] = count2;
  stride_[0] = stride1;
  stride_[1] = stride2;

  int w;
  if (0 == blockLength % 12 && 0 == offset_ % 12) {
    packfn_ = pack_3d<4>;
    unpackfn_ = unpack_3d<4>;
    w = 4;
  } else if (0 == blockLength % 8 && 0 == offset_ % 8) {
    packfn_ = pack_3d<8>;
    unpackfn_ = unpack_3d<8>;
    w = 8;
  } else if (0 == blockLength % 6 && 0 == offset_ % 6) {
    packfn_ = pack_3d<6>;
    unpackfn_ = unpack_3d<6>;
    w = 6;
  } else if (0 == blockLength % 4 && 0 == offset_ % 4) {
    packfn_ = pack_3d<4>;
    unpackfn_ = unpack_3d<4>;
    w = 4;
  } else if (0 == blockLength % 3 && 0 == offset_ % 3) {
    packfn_ = pack_3d<3>;
    unpackfn_ = unpack_3d<3>;
    w = 3;
  } else if (0 == blockLength % 2 && 0 == offset_ % 2) {
    packfn_ = pack_3d<2>;
    unpackfn_ = unpack_3d<2>;
    w = 2;
  } else {
    packfn_ = pack_3d<1>;
    unpackfn_ = unpack_3d<1>;
    w = 1;
  }

  // x dimension is words to load each block
  // y dimension is number of blocks
  // z dimension is object count
  bd_ = Dim3::fill_xyz_by_pow2(Dim3(blockLength / w, count1, count2), 512);
  gd_ = (Dim3(blockLength / w, count1, count2) + bd_ - Dim3(1, 1, 1)) / bd_;
  // gd_ = Dim3((blockLength / w + bd_.x - 1) / bd_.x, (count1 + bd_.y - 1) /
  // bd_.y, (count2 + bd_.z - 1) / bd_.z);

  gd_.y = std::min(int64_t(65535), gd_.y);

  assert(packfn_);
  assert(gd_.x > 0);
  assert(gd_.y > 0);
  assert(gd_.z > 0);
  assert(bd_.x > 0);
  assert(bd_.y > 0);
  assert(bd_.z > 0);
}

void Packer3D::launch_pack(void *outbuf, int *position, const void *inbuf,
                           const int incount, cudaStream_t stream) const {
  TEMPI_COUNTER_OP(pack3d, NUM_PACKS, ++);
  LOG_SPEW("launch_pack offset=" << offset_);
  inbuf = static_cast<const char *>(inbuf) + offset_;
  outbuf = static_cast<char *>(outbuf) + *position;
  packfn_<<<gd_, bd_, 0, stream>>>(outbuf, inbuf, incount, blockLength_,
                                   count_[0], stride_[0], count_[1], stride_[1],
                                   extent_);
  CUDA_RUNTIME(cudaGetLastError());
  assert(position);
  (*position) += incount * count_[1] * count_[0] * blockLength_;
}

void Packer3D::launch_unpack(const void *inbuf, int *position, void *outbuf,
                             const int outcount, cudaStream_t stream) const {
  TEMPI_COUNTER_OP(pack3d, NUM_UNPACKS, ++);
  outbuf = static_cast<char *>(outbuf) + offset_;
  inbuf = static_cast<const char *>(inbuf) + *position;
  unpackfn_<<<gd_, bd_, 0, stream>>>(outbuf, inbuf, outcount, blockLength_,
                                     count_[0], stride_[0], count_[1],
                                     stride_[1], extent_);

  CUDA_RUNTIME(cudaGetLastError());
  assert(position);
  (*position) += outcount * count_[1] * count_[0] * blockLength_;
}

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
