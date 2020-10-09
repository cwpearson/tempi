#include "packer_stride_1.hpp"

#include "cuda_runtime.hpp"
#include "dim3.hpp"
#include "logging.hpp"
#include "streams.hpp"

/* pack blocks of bytes separated by two strides

    z is the count

each thread loads N bytes of a block
 */
template <unsigned N>
__global__ static void pack_bytes_n(
    void *__restrict__ outbuf, int position, const void *__restrict__ inbuf,
    const int incount,
    unsigned blockLength, // block length (B)
    unsigned count,       // count of inner blocks in a group
    unsigned stride       // stride (B) between start of inner blocks in group
) {

  assert(blockLength % N == 0); // N should evenly divide block length

  const unsigned int tz = blockDim.z * blockIdx.z + threadIdx.z;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;

  char *__restrict__ op = reinterpret_cast<char *>(outbuf);
  const char *__restrict__ ip = reinterpret_cast<const char *>(inbuf);

  for (int z = tz; z < gridDim.z * blockDim.z; z += gridDim.z * blockDim.z) {
    char *__restrict__ dst = op + position + z * count * blockLength;
    const char *__restrict__ src = ip + z * stride * count;

    for (unsigned y = ty; y < count; y += gridDim.y * blockDim.y) {
      for (unsigned x = tx; x < blockLength / N; x += gridDim.x * blockDim.x) {
        unsigned bo = y * blockLength + x * N;
        unsigned bi = y * stride + x * N;
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

PackerStride1::PackerStride1(unsigned blockLength, unsigned count,
                             unsigned stride) {
  blockLength_ = blockLength;
  count_ = count;
  stride_ = stride;

  wordSize_ = 1;
  while (0 == blockLength_ % (wordSize_ * 2) && (wordSize_ * 2 <= 4)) {
    wordSize_ *= 2;
  }

  // griddim.z should be incount
  bd_ = Dim3::fill_xyz_by_pow2(Dim3(blockLength_ / wordSize_, count_, 1), 512);
  gd_ = (Dim3(blockLength_ / wordSize_, count_, 1) + bd_ - Dim3(1, 1, 1)) / bd_;
}

void PackerStride1::pack(void *outbuf, int *position, const void *inbuf,
                         const int incount) const {

  int device;
  CUDA_RUNTIME(cudaGetDevice(&device));
  LOG_SPEW("PackerStride1 on CUDA " << device);

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

  Dim3 gd = gd_;
  gd.z = 1;

  if (4 == wordSize_) {
    LOG_SPEW("wordSize_ = 4");
    pack_bytes_n<4><<<gd, bd_, 0, kernStream[device]>>>(
        outbuf, *position, inbuf, incount, blockLength_, count_, stride_);

  } else if (4 == wordSize_) {
    LOG_SPEW("wordSize_ = 4");
    pack_bytes_n<4><<<gd, bd_, 0, kernStream[device]>>>(
        outbuf, *position, inbuf, incount, blockLength_, count_, stride_);

  } else {
    LOG_SPEW("wordSize == 1");
    pack_bytes_n<1><<<gd, bd_, 0, kernStream[device]>>>(
        outbuf, *position, inbuf, incount, blockLength_, count_, stride_);
  }

  CUDA_RUNTIME(cudaGetLastError());

  (*position) += incount * count_ * blockLength_;

  CUDA_RUNTIME(cudaStreamSynchronize(kernStream[device]));
}