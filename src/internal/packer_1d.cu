//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "packer_1d.hpp"

#include "counters.hpp"
#include "cuda_runtime.hpp"
#include "dim3.hpp"
#include "logging.hpp"

Packer1D::Packer1D(unsigned off, unsigned extent)
    : offset_(off), extent_(extent) {}

void Packer1D::launch_pack(void *outbuf, int *position, const void *inbuf,
                           const int incount, cudaStream_t stream) const {
  assert(position);
  assert(outbuf);
  assert(inbuf);
  inbuf = static_cast<const char *>(inbuf) + offset_;
  outbuf = static_cast<char *>(outbuf) + *position;
  const uint64_t nBytes = incount * extent_;
  // cudaMemcpy is not synchronous for d2d
  TEMPI_COUNTER_OP(cudart, MEMCPY_ASYNC_NUM, ++);
  TEMPI_COUNTER_EXPR(double start = MPI_Wtime());
  CUDA_RUNTIME(
      cudaMemcpyAsync(outbuf, inbuf, nBytes, cudaMemcpyDefault, stream));
  TEMPI_COUNTER_OP(cudart, MEMCPY_ASYNC_TIME, += MPI_Wtime() - start);
  (*position) += incount * extent_;
}

void Packer1D::launch_unpack(const void *inbuf, int *position, void *outbuf,
                             const int outcount, cudaStream_t stream) const {
  assert(position);
  assert(outbuf);
  assert(inbuf);
  outbuf = static_cast<char *>(outbuf) + offset_;
  inbuf = static_cast<const char *>(inbuf) + *position;
  const uint64_t nBytes = outcount * extent_;
  // cudaMemcpy is not synchronous for d2d
  TEMPI_COUNTER_OP(cudart, MEMCPY_ASYNC_NUM, ++);
  TEMPI_COUNTER_EXPR(double start = MPI_Wtime());
  CUDA_RUNTIME(
      cudaMemcpyAsync(outbuf, inbuf, nBytes, cudaMemcpyDefault, stream));
  TEMPI_COUNTER_OP(cudart, MEMCPY_ASYNC_TIME, += MPI_Wtime() - start);
  (*position) += outcount * extent_;
}

void Packer1D::pack_async(void *outbuf, int *position, const void *inbuf,
                          const int incount, cudaEvent_t event) const {
  LaunchInfo info = pack_launch_info(inbuf);
  launch_pack(outbuf, position, inbuf, incount, info.stream);
  if (event) {
    TEMPI_COUNTER_OP(cudart, EVENT_RECORD_NUM, ++);
    TEMPI_COUNTER_EXPR(double start = MPI_Wtime());
    CUDA_RUNTIME(cudaEventRecord(event, info.stream));
    TEMPI_COUNTER_OP(cudart, EVENT_RECORD_TIME, += MPI_Wtime() - start);
  }
}

void Packer1D::unpack_async(const void *inbuf, int *position, void *outbuf,
                            const int outcount, cudaEvent_t event) const {
  LaunchInfo info = unpack_launch_info(outbuf);
  launch_unpack(inbuf, position, outbuf, outcount, info.stream);
  if (event) {
    TEMPI_COUNTER_OP(cudart, EVENT_RECORD_NUM, ++);
    TEMPI_COUNTER_EXPR(double start = MPI_Wtime());
    CUDA_RUNTIME(cudaEventRecord(event, info.stream));
    TEMPI_COUNTER_OP(cudart, EVENT_RECORD_TIME, += MPI_Wtime() - start);
  }
}

// same as async but synchronize after launch
void Packer1D::pack(void *outbuf, int *position, const void *inbuf,
                    const int incount) const {
  LaunchInfo info = pack_launch_info(inbuf);
  launch_pack(outbuf, position, inbuf, incount, info.stream);
  TEMPI_COUNTER_OP(cudart, STREAM_SYNC_NUM, ++);
  TEMPI_COUNTER_EXPR(double start = MPI_Wtime());
  CUDA_RUNTIME(cudaStreamSynchronize(info.stream));
  TEMPI_COUNTER_OP(cudart, STREAM_SYNC_TIME, += MPI_Wtime() - start);
}

void Packer1D::unpack(const void *inbuf, int *position, void *outbuf,
                      const int outcount) const {
  LaunchInfo info = unpack_launch_info(outbuf);
  launch_unpack(inbuf, position, outbuf, outcount, info.stream);
  TEMPI_COUNTER_OP(cudart, STREAM_SYNC_NUM, ++);
  TEMPI_COUNTER_EXPR(double start = MPI_Wtime());
  CUDA_RUNTIME(cudaStreamSynchronize(info.stream));
  TEMPI_COUNTER_OP(cudart, STREAM_SYNC_TIME, += MPI_Wtime() - start);
}