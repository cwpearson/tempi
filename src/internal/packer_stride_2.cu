#include "packer_stride_2.hpp"

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
    unsigned blockLength, // block length (B)
    unsigned count0,      // count of inner blocks in a group
    unsigned stride0,     // stride (B) between start of inner blocks in group
    unsigned count1,      // number of block groups
    unsigned stride1      // stride (B) between start of block groups
) {

  assert(blockLength % N == 0); // N should evenly divide block length

#if 0
  printf("count0=%u count1=%u, stride0=%u stride1=%u\n", count0, count1,
         stride0, stride1);
#endif
  // n-1 counts of the stride, plus the extent of the last count
  const int extent =
      (count1 - 1) * stride1 + (count0 - 1) * stride0 + blockLength;

  const unsigned int tz = blockDim.z * blockIdx.z + threadIdx.z;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;

  char *__restrict__ op = reinterpret_cast<char *>(outbuf) + position;
  const char *__restrict__ ip = reinterpret_cast<const char *>(inbuf);

  for (int i = 0; i < incount; ++i) {
    char *__restrict__ dst = op + i * count1 * count0 * blockLength;
    const char *__restrict__ src = ip + i * extent;

    for (unsigned z = tz; z < count1; z += gridDim.z * blockDim.z) {
      for (unsigned y = ty; y < count0; y += gridDim.y * blockDim.y) {
        for (unsigned x = tx; x < blockLength / N;
             x += gridDim.x * blockDim.x) {
          unsigned bo = z * count0 * blockLength + y * blockLength + x * N;
          unsigned bi = z * stride1 + y * stride0 + x * N;
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
    const unsigned blockLength, // block length (B)
    const unsigned count0,      // count of inner blocks in a group
    const unsigned stride0, // stride (B) between start of inner blocks in group
    const unsigned count1,  // number of block groups
    const unsigned stride1  // stride (B) between start of block groups
) {

  assert(blockLength % N == 0); // N should evenly divide block length

  // n-1 counts of the stride, plus the extent of the last count
  const int extent =
      (count1 - 1) * stride1 + (count0 - 1) * stride0 + blockLength;

  const unsigned int tz = blockDim.z * blockIdx.z + threadIdx.z;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;

  char *__restrict__ op = reinterpret_cast<char *>(outbuf) + position;
  const char *__restrict__ ip = reinterpret_cast<const char *>(inbuf);

  for (int i = 0; i < incount; ++i) {
    char *__restrict__ dst = op + i * extent;
    const char *__restrict__ src = ip + i * count1 * count0 * blockLength;

    for (unsigned z = tz; z < count1; z += gridDim.z * blockDim.z) {
      for (unsigned y = ty; y < count0; y += gridDim.y * blockDim.y) {
        for (unsigned x = tx; x < blockLength / N;
             x += gridDim.x * blockDim.x) {
          unsigned bi = z * count0 * blockLength + y * blockLength + x * N;
          unsigned bo = z * stride1 + y * stride0 + x * N;
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

PackerStride2::PackerStride2(unsigned blockLength, unsigned count0,
                             unsigned stride0, unsigned count1,
                             unsigned stride1) {
  assert(blockLength_ > 0);
  blockLength_ = blockLength;
  count_[0] = count0;
  count_[1] = count1;
  stride_[0] = stride0;
  stride_[1] = stride1;

  wordSize_ = 1;
  while (0 == blockLength % (wordSize_ * 2) && (wordSize_ * 2 <= 4)) {
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

void PackerStride2::pack_async(void *outbuf, int *position, const void *inbuf,
                               const int incount) const {

  int device;
  CUDA_RUNTIME(cudaGetDevice(&device));
  LaunchInfo info = pack_launch_info(inbuf);
  LOG_SPEW("PackerStride2::pack on CUDA " << info.device);
  CUDA_RUNTIME(cudaSetDevice(info.device));

#if 0
  char *__restrict__ op = reinterpret_cast<char *>(outbuf);
  const char *__restrict__ ip = reinterpret_cast<const char *>(inbuf);

  for (int i = 0; i < incount; ++i) {
    char *__restrict__ dst =
        op + *position + i * count_[1] * count_[0] * blockLength_;
    const char *__restrict__ src =
        ip + i * stride_[1] * count_[1] * stride_[0] * count_[0];

    for (unsigned z = 0; z < count_[1]; ++z) {
      for (unsigned y = 0; y < count_[0]; ++y) {
        for (unsigned x = 0; x < blockLength_; ++x) {
          int64_t bo = z * count_[0] * blockLength_ + y * blockLength_ + x;
          int64_t bi = z * stride_[1] + y * stride_[0] + x;
          // std::cerr << bi << " -> " << bo << "\n";
          dst[bo] = src[bi];
        }
      }
    }
  }
#endif

  if (4 == wordSize_) {
    LOG_SPEW("wordSize_ = 4");
    pack_bytes<4><<<gd_, bd_, 0, info.stream>>>(
        outbuf, *position, inbuf, incount, blockLength_, count_[0], stride_[0],
        count_[1], stride_[1]);

  } else if (8 == wordSize_) {
    LOG_SPEW("wordSize_ = 8");
    pack_bytes<8><<<gd_, bd_, 0, info.stream>>>(
        outbuf, *position, inbuf, incount, blockLength_, count_[0], stride_[0],
        count_[1], stride_[1]);

  } else {
    LOG_SPEW("wordSize == 1");
    pack_bytes<1><<<gd_, bd_, 0, info.stream>>>(
        outbuf, *position, inbuf, incount, blockLength_, count_[0], stride_[0],
        count_[1], stride_[1]);
  }

  CUDA_RUNTIME(cudaGetLastError());

  assert(position);
  (*position) += incount * count_[1] * count_[0] * blockLength_;

  LOG_SPEW("PackerStride2::restore device " << device);
  CUDA_RUNTIME(cudaSetDevice(device));
}

void PackerStride2::unpack(const void *inbuf, int *position, void *outbuf,
                           const int outcount) const {

  int device;
  CUDA_RUNTIME(cudaGetDevice(&device));
  LaunchInfo info = unpack_launch_info(outbuf);
  LOG_SPEW("PackerStride2::unpack on CUDA " << info.device);
  CUDA_RUNTIME(cudaSetDevice(info.device));

  if (4 == wordSize_) {
    LOG_SPEW("wordSize_ = 4");
    unpack_bytes<4><<<gd_, bd_, 0, info.stream>>>(
        outbuf, *position, inbuf, outcount, blockLength_, count_[0], stride_[0],
        count_[1], stride_[1]);

  } else if (8 == wordSize_) {
    LOG_SPEW("wordSize_ = 8");
    unpack_bytes<8><<<gd_, bd_, 0, info.stream>>>(
        outbuf, *position, inbuf, outcount, blockLength_, count_[0], stride_[0],
        count_[1], stride_[1]);

  } else {
    LOG_SPEW("wordSize == 1");
    unpack_bytes<1><<<gd_, bd_, 0, info.stream>>>(
        outbuf, *position, inbuf, outcount, blockLength_, count_[0], stride_[0],
        count_[1], stride_[1]);
  }

  CUDA_RUNTIME(cudaGetLastError());

  (*position) += outcount * count_[1] * count_[0] * blockLength_;

  CUDA_RUNTIME(cudaStreamSynchronize(info.stream));
  LOG_SPEW("PackerStride2::restore device " << device);
  CUDA_RUNTIME(cudaSetDevice(device));
}
