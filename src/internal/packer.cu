#include "packer.hpp"

#include "cuda_runtime.hpp"
#include "dim3.hpp"
#include "logging.hpp"
#include "streams.hpp"

__global__ static void pack_bytes(void *__restrict__ outbuf, int position,
                                  const void *__restrict__ inbuf,
                                  const int incount, unsigned blockLength,
                                  unsigned count0, unsigned stride0,
                                  unsigned count1, unsigned stride1) {

  const unsigned int tz = blockDim.z * blockIdx.z + threadIdx.z;
  const unsigned int ty = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int tx = blockDim.x * blockIdx.x + threadIdx.x;

  char *__restrict__ op = reinterpret_cast<char *>(outbuf);
  const char *__restrict__ ip = reinterpret_cast<const char *>(inbuf);

  for (int i = 0; i < incount; ++i) {
    char *__restrict__ dst = op + position + i * count1 * count0 * blockLength;
    const char *__restrict__ src = ip + i * stride1 * count1 * stride0 * count0;

    for (unsigned z = tz; z < count1; z += gridDim.z * blockDim.z) {
      for (unsigned y = ty; y < count0; y += gridDim.y * blockDim.y) {
        for (unsigned x = tx; x < blockLength; x += gridDim.x * blockDim.x) {
          unsigned bo = z * count0 * blockLength + y * blockLength + x;
          unsigned bi = z * stride1 + y * stride0 + x;
          // printf("%u -> %u\n", bi, bo);
          dst[bo] = src[bi];
        }
      }
    }
  }
}

PackerStride2::PackerStride2(unsigned blockLength, unsigned count0,
                             unsigned stride0, unsigned count1,
                             unsigned stride1) {
  blockLength_ = blockLength;
  count_[0] = count0;
  count_[1] = count1;
  stride_[0] = stride0;
  stride_[1] = stride1;
}

void PackerStride2::pack(void *outbuf, int *position, const void *inbuf,
                         const int incount) const {

  int device;
  CUDA_RUNTIME(cudaGetDevice(&device));
  LOG_SPEW("PackerStride2 on CUDA " << device);

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

  const Dim3 bd =
      Dim3::fill_xyz_by_pow2(Dim3(blockLength_, count_[0], count_[1]), 512);
  const Dim3 gd =
      (Dim3(blockLength_, count_[0], count_[1]) + bd - Dim3(1, 1, 1)) / bd;

  pack_bytes<<<gd, bd, 0, kernStream[device]>>>(
      outbuf, *position, inbuf, incount, blockLength_, count_[0], stride_[0],
      count_[1], stride_[1]);
  CUDA_RUNTIME(cudaGetLastError());

  (*position) += incount * count_[1] * count_[0] * blockLength_;

  CUDA_RUNTIME(cudaStreamSynchronize(kernStream[device]));
}