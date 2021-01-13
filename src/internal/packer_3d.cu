//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

/* TODO: alignment is not really handled in this case.
   we expect all access to be aligned
*/

#include "packer_3d.hpp"

#include "cuda_runtime.hpp"
#include "dim3.hpp"
#include "logging.hpp"
#include "streams.hpp"

/* pack blocks of bytes separated by two strides

each thread loads N bytes of a block
 */
template <unsigned N>
__global__ static void pack_bytes(
    void *__restrict__ outbuf, int position, // position in output buffer
    const void *__restrict__ inbuf, const int incount,
    unsigned count0,  // block length (B)
    unsigned count1,  // count of inner blocks in a group
    unsigned stride1, // stride (B) between start of inner blocks in group
    unsigned count2,  // number of block groups
    unsigned stride2  // stride (B) between start of block groups
) {

  assert(count0 % N == 0); // N should evenly divide block length

#if 0
  printf("count1=%u count2=%u, stride1=%u stride2=%u\n", count1, count2,
         stride1, stride2);
#endif
  // n-1 counts of the stride, plus the extent of the last count
  const uint64_t extent =
      (count2 - 1) * stride2 + (count1 - 1) * stride1 + count0;

  const unsigned int tz = blockDim.z * blockIdx.z + threadIdx.z;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;

  char *__restrict__ op = reinterpret_cast<char *>(outbuf) + position;
  const char *__restrict__ ip = reinterpret_cast<const char *>(inbuf);

  for (int i = 0; i < incount; ++i) {
    char *__restrict__ dst = op + i * count2 * count1 * count0;
    const char *__restrict__ src = ip + i * extent;

    for (unsigned z = tz; z < count2; z += gridDim.z * blockDim.z) {
      for (unsigned y = ty; y < count1; y += gridDim.y * blockDim.y) {
        for (unsigned x = tx; x < count0 / N; x += gridDim.x * blockDim.x) {
          unsigned bo = z * count1 * count0 + y * count0 + x * N;
          unsigned bi = z * stride2 + y * stride1 + x * N;
#if 0
          printf("%lu -> %lu\n", uintptr_t(src) + bi - uintptr_t(inbuf),
                 uintptr_t(dst) + bo - uintptr_t(outbuf));
#endif

          if (N == 1) {
            dst[bo] = src[bi];
          } else if (N == 2) {
            uint16_t *__restrict__ d = reinterpret_cast<uint16_t *>(dst + bo);
            const uint16_t *__restrict__ s =
                reinterpret_cast<const uint16_t *>(src + bi);
            *d = *s;
          } else if (N == 4) {
            uint32_t *__restrict__ d = reinterpret_cast<uint32_t *>(dst + bo);
            const uint32_t *__restrict__ s =
                reinterpret_cast<const uint32_t *>(src + bi);
            *d = *s;
          } else if (N == 8) {
            uint64_t *__restrict__ d = reinterpret_cast<uint64_t *>(dst + bo);
            const uint64_t *__restrict__ s =
                reinterpret_cast<const uint64_t *>(src + bi);
            *d = *s;
          }
        }
      }
    }
  }
}

template <unsigned N>
__global__ static void unpack_bytes(
    void *__restrict__ outbuf, int position, const void *__restrict__ inbuf,
    const int incount,
    const unsigned count0,  // block length (B)
    const unsigned count1,  // count of inner blocks in a group
    const unsigned stride1, // stride (B) between start of inner blocks in group
    const unsigned count2,  // number of block groups
    const unsigned stride2  // stride (B) between start of block groups
) {

  assert(count0 % N == 0); // N should evenly divide block length

  // n-1 counts of the stride, plus the extent of the last count
  const uint64_t extent =
      (count2 - 1) * stride2 + (count1 - 1) * stride1 + count0;

  const unsigned int tz = blockDim.z * blockIdx.z + threadIdx.z;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;

  char *__restrict__ op = reinterpret_cast<char *>(outbuf) + position;
  const char *__restrict__ ip = reinterpret_cast<const char *>(inbuf);

  for (int i = 0; i < incount; ++i) {
    char *__restrict__ dst = op + i * extent;
    const char *__restrict__ src = ip + i * count2 * count1 * count0;

    for (unsigned z = tz; z < count2; z += gridDim.z * blockDim.z) {
      for (unsigned y = ty; y < count1; y += gridDim.y * blockDim.y) {
        for (unsigned x = tx; x < count0 / N; x += gridDim.x * blockDim.x) {
          unsigned bi = z * count1 * count0 + y * count0 + x * N;
          unsigned bo = z * stride2 + y * stride1 + x * N;
          // printf("%u -> %u\n", bi, bo);

          if (N == 1) {
            dst[bo] = src[bi];
          } else if (N == 2) {
            uint16_t *__restrict__ d = reinterpret_cast<uint16_t *>(dst + bo);
            const uint16_t *__restrict__ s =
                reinterpret_cast<const uint16_t *>(src + bi);
            *d = *s;
          } else if (N == 4) {
            uint32_t *__restrict__ d = reinterpret_cast<uint32_t *>(dst + bo);
            const uint32_t *__restrict__ s =
                reinterpret_cast<const uint32_t *>(src + bi);
            *d = *s;
          } else if (N == 8) {
            uint64_t *__restrict__ d = reinterpret_cast<uint64_t *>(dst + bo);
            const uint64_t *__restrict__ s =
                reinterpret_cast<const uint64_t *>(src + bi);
            *d = *s;
          }
        }
      }
    }
  }
}

Packer3D::Packer3D(unsigned off, unsigned blockLength, unsigned count1,
                   unsigned stride1, unsigned count2, unsigned stride2) {
  offset_ = off;
  blockLength_ = blockLength;
  assert(blockLength_ > 0);
  count_[0] = count1;
  count_[1] = count2;
  stride_[0] = stride1;
  stride_[1] = stride2;

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

  // bd_ = Dim3(1, 1, 1);
  // gd_ = Dim3(1, 1, 1);
}

void Packer3D::launch_pack(void *outbuf, int *position, const void *inbuf,
                           const int incount, cudaStream_t stream) const {

  LOG_SPEW("launch_pack offset=" << offset_);
  inbuf = static_cast<const char *>(inbuf) + offset_;

  if (uintptr_t(inbuf) % wordSize_) {
    LOG_WARN("pack kernel may be unaligned.");
  }

  if (4 == wordSize_) {
    LOG_SPEW("wordSize_ = 4");

    pack_bytes<4><<<gd_, bd_, 0, stream>>>(outbuf, *position, inbuf, incount,
                                           blockLength_, count_[0], stride_[0],
                                           count_[1], stride_[1]);

  } else if (8 == wordSize_) {
    LOG_SPEW("wordSize_ = 8");
    pack_bytes<8><<<gd_, bd_, 0, stream>>>(outbuf, *position, inbuf, incount,
                                           blockLength_, count_[0], stride_[0],
                                           count_[1], stride_[1]);

  } else {
    LOG_SPEW("wordSize == 1");
    pack_bytes<1><<<gd_, bd_, 0, stream>>>(outbuf, *position, inbuf, incount,
                                           blockLength_, count_[0], stride_[0],
                                           count_[1], stride_[1]);
  }

  CUDA_RUNTIME(cudaGetLastError());
  assert(position);
  (*position) += incount * count_[1] * count_[0] * blockLength_;
}

void Packer3D::launch_unpack(const void *inbuf, int *position, void *outbuf,
                             const int outcount, cudaStream_t stream) const {

  outbuf = static_cast<char *>(outbuf) + offset_;

  if (4 == wordSize_) {
    LOG_SPEW("wordSize_ = 4");
    unpack_bytes<4><<<gd_, bd_, 0, stream>>>(outbuf, *position, inbuf, outcount,
                                             blockLength_, count_[0],
                                             stride_[0], count_[1], stride_[1]);

  } else if (8 == wordSize_) {
    LOG_SPEW("wordSize_ = 8");
    unpack_bytes<8><<<gd_, bd_, 0, stream>>>(outbuf, *position, inbuf, outcount,
                                             blockLength_, count_[0],
                                             stride_[0], count_[1], stride_[1]);

  } else {
    LOG_SPEW("wordSize == 1");
    unpack_bytes<1><<<gd_, bd_, 0, stream>>>(outbuf, *position, inbuf, outcount,
                                             blockLength_, count_[0],
                                             stride_[0], count_[1], stride_[1]);
  }

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
                          const int incount) const {
  LaunchInfo info = pack_launch_info(inbuf);
  launch_pack(outbuf, position, inbuf, incount, info.stream);
}

// same as async but synchronize after launch
void Packer3D::pack(void *outbuf, int *position, const void *inbuf,
                    const int incount) const {
  LaunchInfo info = pack_launch_info(inbuf);
  launch_pack(outbuf, position, inbuf, incount, info.stream);
  CUDA_RUNTIME(cudaStreamSynchronize(info.stream));
}

#if 0
void Packer3D::unpack_async(const void *inbuf, int *position, void *outbuf,
                           const int outcount) const {

  int device;
  CUDA_RUNTIME(cudaGetDevice(&device));
  LaunchInfo info = unpack_launch_info(outbuf);
  LOG_SPEW("Packer3D::unpack on CUDA " << info.device);
  CUDA_RUNTIME(cudaSetDevice(info.device));
  launch_unpack(inbuf, position, outbuf, outcount, info.stream);
  LOG_SPEW("Packer3D::restore device " << device);
  CUDA_RUNTIME(cudaSetDevice(device));
}
#endif

void Packer3D::unpack(const void *inbuf, int *position, void *outbuf,
                      const int outcount) const {
  LaunchInfo info = unpack_launch_info(outbuf);
  launch_unpack(inbuf, position, outbuf, outcount, info.stream);
  CUDA_RUNTIME(cudaStreamSynchronize(info.stream));
}
