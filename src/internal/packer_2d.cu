//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "packer_2d.hpp"

#include "cuda_runtime.hpp"
#include "dim3.hpp"
#include "logging.hpp"

/* pack blocks of bytes separated a stride

    the z dimension is used for the incount

each thread loads N bytes of a block
 */
template <unsigned N>
__global__ static void
pack_bytes(void *__restrict__ outbuf,
           int position, // location in the output buffer to start packing (B)
           const void *__restrict__ inbuf,
           const int incount, // number of datatypes to pack
           unsigned count0,   // bytes in dim 0
           unsigned count1,   // elements in dim 1
           unsigned stride1   // stride (B) between elements in dim1
                              // stride0 is implicitly 1
) {

  assert(count0 % N == 0); // N should evenly divide block length
  assert(count1 >= 1);

  // as the input space may be large, incount * extent may be over 2G
  const uint64_t extent = (count1 - 1) * stride1 + count0;

  const unsigned int tz = blockDim.z * blockIdx.z + threadIdx.z;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;

  char *__restrict__ op = reinterpret_cast<char *>(outbuf) + position;
  const char *__restrict__ ip = reinterpret_cast<const char *>(inbuf);

  for (int z = tz; z < incount; z += gridDim.z * blockDim.z) {
    // each packed datatype will take count1 * count0 bytes in outbuf
    char *__restrict__ dst = op + z * count1 * count0;
    // each datatype input has extent separating their starts
    const char *__restrict__ src = ip + z * extent;

    // if (tz == 0 && ty == 0 && tx == 0) {
    //   printf("src offset =%d\n", z * extent);
    // }

    // x direction handle the blocks, y handles the block counts
    for (unsigned y = ty; y < count1; y += gridDim.y * blockDim.y) {
      for (unsigned x = tx; x < count0 / N; x += gridDim.x * blockDim.x) {
        unsigned bo = y * count0 + x * N;
        unsigned bi = y * stride1 + x * N;
        // printf("%u -> %u\n", bi, bo);

#if 0
        {
          uintptr_t ioff = uintptr_t(src + bi) - uintptr_t(inbuf);
          uintptr_t ooff = uintptr_t(dst + bo) - uintptr_t(outbuf);
          if (ioff >= 4294963208ull) {
            printf("ioff=%lu bi=%u, z=%d, z*ext=%lu\n", ioff, bi, z, z*extent);
          }
          if (ooff >= 8388608) {
            printf("ooff=%lu bo=%u, z=%d, z*bl*cnt=%d\n", ooff, bo, z, z * count0 * count);
          }
        }
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

/* unpack

    griddim.z is the count

each thread loads N bytes of a block
*/
template <unsigned N>
__global__ static void unpack_bytes(
    void *__restrict__ outbuf,
    int position, // location in the output buffer to start unpacking (B)
    const void *__restrict__ inbuf,
    const int outcount,   // number of datatypes to unpack
    unsigned blockLength, // block length (B)
    unsigned count,       // count of blocks in a group
    unsigned stride       // stride (B) between start of blocks in group
) {

  assert(blockLength % N == 0); // N should evenly divide block length

  const uint64_t extent = (count - 1) * stride + blockLength;

  const unsigned int tz = blockDim.z * blockIdx.z + threadIdx.z;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;

  char *__restrict__ op = reinterpret_cast<char *>(outbuf) + position;
  const char *__restrict__ ip = reinterpret_cast<const char *>(inbuf);

  for (int z = tz; z < outcount; z += gridDim.z * blockDim.z) {
    // each datatype will have stride * count separating their starts in outbuf
    // each packed datatype has blockLength * count separating their starts
    char *__restrict__ dst = op + z * extent;
    const char *__restrict__ src = ip + z * blockLength * count;

    // x direction handle the blocks, y handles the block counts
    for (unsigned y = ty; y < count; y += gridDim.y * blockDim.y) {
      for (unsigned x = tx; x < blockLength / N; x += gridDim.x * blockDim.x) {
        unsigned bi = y * blockLength + x * N;
        unsigned bo = y * stride + x * N;
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

Packer2D::Packer2D(unsigned off, unsigned blockLength, unsigned count,
                   unsigned stride) {
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
  inbuf = static_cast<const char *>(inbuf) + offset_;

  if (uintptr_t(inbuf) % wordSize_) {
    LOG_WARN("pack kernel may be unaligned.");
  }

  Dim3 gd = gd_;
  gd.z = incount;
  // LOG_SPEW("wordSize_ = " << wordSize_);
  if (kernelStart) {
    CUDA_RUNTIME(cudaEventRecord(kernelStart, stream));
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
                          const int incount) const {
  LaunchInfo info = pack_launch_info(inbuf);
  launch_pack(outbuf, position, inbuf, incount, info.stream);
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