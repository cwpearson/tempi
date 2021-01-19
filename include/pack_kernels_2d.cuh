//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "dim3.hpp"
#include "pack_config.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>

/* collaboratively use x dimension to copy `n` bytes with word size `W`
 */
template <unsigned W>
__device__ void grid_x_memcpy_aligned(void *__restrict__ dst,
                                      const void *__restrict__ src, size_t n) {

  static_assert(sizeof(uchar3) == 3, "wrong uchar3 size");
  static_assert(sizeof(ushort3) == 6, "wrong ushort3 size");
  static_assert(sizeof(uint3) == 12, "wrong uint3 size");
  static_assert(sizeof(ulonglong2) == 16, "wrong ulonglong2 size");
  static_assert(sizeof(ulonglong3) == 24, "wrong ulonglong3 size");
  static_assert(sizeof(ulonglong4) == 32, "wrong ulonglong4 size");

  assert(n % W == 0 && "wrong word size");

  int tx = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = tx; i < n / W; i += gridDim.x * blockDim.x) {
    if (2 == W) {
      static_cast<uint16_t *>(dst)[i] = static_cast<const uint16_t *>(src)[i];
    } else if (3 == W) {
      static_cast<uchar3 *>(dst)[i] = static_cast<const uchar3 *>(src)[i];
    } else if (4 == W) {
      static_cast<uint32_t *>(dst)[i] = static_cast<const uint32_t *>(src)[i];
    } else if (6 == W) {
      static_cast<ushort3 *>(dst)[i] = static_cast<const ushort3 *>(src)[i];
    } else if (8 == W) {
      static_cast<uint64_t *>(dst)[i] = static_cast<const uint64_t *>(src)[i];
    } else if (12 == W) { // this kills performance in zero-copy
      static_cast<uint3 *>(dst)[i] = static_cast<const uint3 *>(src)[i];
    } else if (16 == W) {
      static_cast<ulonglong2 *>(dst)[i] =
          static_cast<const ulonglong2 *>(src)[i];
    } else if (24 == W) {
      static_cast<ulonglong3 *>(dst)[i] =
          static_cast<const ulonglong3 *>(src)[i];
    } else if (32 == W) {
      static_cast<ulonglong4 *>(dst)[i] =
          static_cast<const ulonglong4 *>(src)[i];
    } else { // default to W == 1
      static_cast<uint8_t *>(dst)[i] = static_cast<const uint8_t *>(src)[i];
    }
  }
}

/* use one warp to copy each contiguous block of bytes
 */
template <unsigned W>
__global__ void pack_bytes_warp(void *__restrict__ outbuf,
                                const void *__restrict__ inbuf,
                                const unsigned incount, const unsigned count0,
                                const unsigned count1, const unsigned stride1) {

  // as the input space may be large, incount * extent may be over 2G
  // FIXME: this may not handle vector vs subarray correctly
  // may need to compute outside kernel and pass in
  const uint64_t extent = (count1 - 1) * stride1 + count0;

  const unsigned int tz = blockDim.z * blockIdx.z + threadIdx.z;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;

  assert(blockDim.z == 1);
  for (int z = tz; z < incount; z += gridDim.z) {
    char *__restrict__ dst =
        reinterpret_cast<char *>(outbuf) + z * count1 * count0;
    const char *__restrict__ src =
        reinterpret_cast<const char *>(inbuf) + z * extent;

    for (unsigned y = ty; y < count1; y += gridDim.y * blockDim.y) {
      unsigned bo = y * count0;
      unsigned bi = y * stride1;
#if 0
      if (tx == 0) {
        printf("%u %u\n", bi, count0);
      }
#endif
      //   warp_memcpy_aligned<W>(dst + bo, src + bi, count0);
      grid_x_memcpy_aligned<W>(dst + bo, src + bi, count0);
    }
  }
}

PackConfig::PackConfig(unsigned blockLength, unsigned blockCount)
    : packfn(nullptr) {

  int w;
  /* using largest sizes can reduce the zero-copy performanc especially
  need to detect 12 so it doesn't match 6
  using the W=12 specialization leads to bad zero-copy performance
    */
  if (0 == blockLength % 12) {
    w = 4;
    packfn = pack_bytes_warp<4>;
  } else if (0 == blockLength % 8) {
    w = 8;
    packfn = pack_bytes_warp<8>;
  } else if (0 == blockLength % 6) {
    packfn = pack_bytes_warp<6>;
    w = 6;
  } else if (0 == blockLength % 4) {
    packfn = pack_bytes_warp<4>;
    w = 4;
  } else if (0 == blockLength % 3) {
    packfn = pack_bytes_warp<3>;
    w = 3;
  } else if (0 == blockLength % 2) {
    packfn = pack_bytes_warp<2>;
    w = 2;
  } else {
    packfn = pack_bytes_warp<1>;
    w = 1;
  }

  // one warp in the x dimension
  // y dimension is number of blocks
  // z dimension is object count
  dimBlock = Dim3::fill_xyz_by_pow2(Dim3(blockLength / w, blockCount, 1), 512);
  dimGrid = Dim3((blockLength / w + dimBlock.x - 1) / dimBlock.x,
                 (blockCount + dimBlock.y - 1) / dimBlock.y,
                 0 /* to be filled in dim_grid */);

  dimGrid.y = std::min(65535u, dimGrid.y);

  assert(packfn);
  assert(dimGrid.x > 0);
  assert(dimGrid.y > 0);
  assert(dimGrid.z > 0);
  assert(dimBlock.x > 0);
  assert(dimBlock.y > 0);
  assert(dimBlock.z > 0);
}

// update the grid dimension for `count` objects
dim3 PackConfig::dim_grid(int count) const {
  dim3 ret = dimGrid;
  ret.z = count;
  return ret;
}

dim3 PackConfig::dim_block() const { return dimBlock; }

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

    gridDim.z handles the count

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