//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "measure_system.hpp"

#include "allocators.hpp"
#include "benchmark.hpp"
#include "cuda_runtime.hpp"
#include "logging.hpp"
#include "numeric.hpp"
#include "packer_2d.hpp"
#include "symbols.hpp"
#include "topology.hpp"

#include <mpi.h>

#include <chrono>
#include <iostream>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::time_point<Clock, Duration> Time;

/*extern*/ SystemPerformance systemPerformance;

void measure_system_init() { import_system_performance(systemPerformance); }



static __global__ void kernel(int *a) {
  if (a) {
    *a = threadIdx.x;
  }
}

class KernelLaunchBenchmark : public Benchmark {
  cudaStream_t stream;

public:
  KernelLaunchBenchmark() { CUDA_RUNTIME(cudaStreamCreate(&stream)); }

  ~KernelLaunchBenchmark() { CUDA_RUNTIME(cudaStreamDestroy(stream)); }

  Benchmark::Sample run_iter() override {
    Sample res{};

    Time start = Clock::now();
    for (int i = 0; i < 32; ++i) {
      kernel<<<1, 1, 0, stream>>>(nullptr);
    }
    Time stop = Clock::now();
    Duration dur = stop - start;
    CUDA_RUNTIME(cudaStreamSynchronize(stream));

    res.time = dur.count() / 32.0;

    return res;
  }
};

/* The time to call CudaMemcpyAsync
 */
class CudaMemcpyAsync : public Benchmark {
  cudaStream_t stream;

public:
  CudaMemcpyAsync() { CUDA_RUNTIME(cudaStreamCreate(&stream)); }

  ~CudaMemcpyAsync() { CUDA_RUNTIME(cudaStreamDestroy(stream)); }

  Benchmark::Sample run_iter() override {
    Sample res{};

    Time start = Clock::now();
    for (int i = 0; i < 32; ++i) {
      kernel<<<1, 1, 0, stream>>>(nullptr);
    }
    Time stop = Clock::now();
    Duration dur = stop - start;
    CUDA_RUNTIME(cudaStreamSynchronize(stream));

    res.time = dur.count() / 32.0;

    return res;
  }
};

class CudaMemcpyAsyncD2H : public Benchmark {
  cudaStream_t stream;
  char *src, *dst;
  size_t n_;

public:
  CudaMemcpyAsyncD2H(size_t n) : n_(n) {
    CUDA_RUNTIME(cudaStreamCreate(&stream));
    CUDA_RUNTIME(cudaMalloc(&src, n));
    dst = hostAllocator.allocate(n);
  }

  ~CudaMemcpyAsyncD2H() {
    CUDA_RUNTIME(cudaStreamDestroy(stream));
    CUDA_RUNTIME(cudaFree(src));
    hostAllocator.deallocate(dst, n_);
  }

  void setup() {
    nreps_ = 1;
    run_iter();                       // warmup
    Benchmark::Sample s = run_iter(); // measure time

    // target at least 200us
    nreps_ = 200.0 * 1e-6 / s.time;
    nreps_ = std::max(nreps_, 1);
    LOG_DEBUG("estimate nreps_=" << nreps_ << " for 200us");
  }

  Benchmark::Sample run_iter() override {
    Sample res{};
    Time start = Clock::now();
    for (int i = 0; i < nreps_; ++i) {
      cudaMemcpyAsync(dst, src, n_, cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
    }
    Time stop = Clock::now();
    CUDA_RUNTIME(cudaGetLastError());
    Duration dur = stop - start;
    res.time = dur.count() / double(nreps_);
    return res;
  }
};

class CudaMemcpyAsyncH2D : public Benchmark {
  cudaStream_t stream;
  char *src, *dst;
  size_t n_;

public:
  CudaMemcpyAsyncH2D(size_t n) : n_(n) {
    CUDA_RUNTIME(cudaStreamCreate(&stream));
    CUDA_RUNTIME(cudaMalloc(&dst, n));
    src = hostAllocator.allocate(n);
    // std::memset(src, -1, n); // cache host buffer
  }

  ~CudaMemcpyAsyncH2D() {
    CUDA_RUNTIME(cudaStreamDestroy(stream));
    CUDA_RUNTIME(cudaFree(dst));
    hostAllocator.deallocate(src, n_);
  }

  Benchmark::Sample run_iter() override {
    Sample res{};
    Time start = Clock::now();
    const int nreps = 10;
    for (int i = 0; i < nreps; ++i) {
      cudaMemcpyAsync(dst, src, n_, cudaMemcpyHostToDevice, stream);
      cudaStreamSynchronize(stream);
    }
    Time stop = Clock::now();
    CUDA_RUNTIME(cudaGetLastError());
    Duration dur = stop - start;
    res.time = dur.count() / double(nreps);
    return res;
  }
};

/* a cpu-to-cpu MPI ping-pong test between ranks 0 and 1
   in TEMPI, this is used with the hostAllocator
 */
class CpuCpuPingpong : public MpiBenchmark {
  char *buf_;
  int n_;

public:
  // zero buffer to put it in cache
  CpuCpuPingpong(size_t n, MPI_Comm comm)
      : buf_(nullptr), n_(n), MpiBenchmark(comm) {
    buf_ = hostAllocator.allocate(n);
  }
  ~CpuCpuPingpong() { hostAllocator.deallocate(buf_, n_); }

  Benchmark::Sample run_iter() override {
    int rank;
    MPI_Comm_rank(comm_, &rank);
    MPI_Barrier(comm_);
    Time start = Clock::now();
    for (int i = 0; i < nreps_; ++i) {
      if (0 == rank) {
        libmpi.MPI_Send(buf_, n_, MPI_BYTE, 1, 0, comm_);
        libmpi.MPI_Recv(buf_, n_, MPI_BYTE, 1, 0, comm_, MPI_STATUS_IGNORE);
      } else if (1 == rank) {
        libmpi.MPI_Recv(buf_, n_, MPI_BYTE, 0, 0, comm_, MPI_STATUS_IGNORE);
        libmpi.MPI_Send(buf_, n_, MPI_BYTE, 0, 0, comm_);
      }
    }
    Time stop = Clock::now();
    Duration dur = stop - start;

    double time = dur.count() / 2, maxTime;
    MPI_Allreduce(&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, comm_);

    Sample res{};
    res.time = maxTime / double(nreps_);

    return res;
  }
};

/* a gpu-to-gpu MPI ping-pong test
 */
class GpuGpuPingpong : public MpiBenchmark {
  char *buf_;
  size_t n_;

public:
  GpuGpuPingpong(size_t n, MPI_Comm comm) : n_(n), MpiBenchmark(comm) {
    CUDA_RUNTIME(cudaMalloc(&buf_, n));
  }
  ~GpuGpuPingpong() { CUDA_RUNTIME(cudaFree(buf_)); }

  Benchmark::Sample run_iter() override {

    int rank;
    MPI_Comm_rank(comm_, &rank);
    MPI_Barrier(comm_);
    Time start = Clock::now();
    for (int i = 0; i < nreps_; ++i) {

      if (0 == rank) {
        libmpi.MPI_Send(buf_, n_, MPI_BYTE, 1, 0, comm_);
        libmpi.MPI_Recv(buf_, n_, MPI_BYTE, 1, 0, comm_, MPI_STATUS_IGNORE);
      } else if (1 == rank) {
        libmpi.MPI_Recv(buf_, n_, MPI_BYTE, 0, 0, comm_, MPI_STATUS_IGNORE);
        libmpi.MPI_Send(buf_, n_, MPI_BYTE, 0, 0, comm_);
      }
    }
    Time stop = Clock::now();
    Duration dur = stop - start;

    double time = dur.count() / 2, maxTime;
    MPI_Allreduce(&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, comm_);

    Sample res{};
    res.time = maxTime / double(nreps_);

    return res;
  }
};

/* Synchronous pack, as used in MPI_Send
 */
class DevicePack2D : public Benchmark {
  char *src, *dst;
  int64_t numBlocks_;
  int64_t blockLength_;
  int64_t stride_;
  Packer2D packer_;

public:
  DevicePack2D(int64_t numBlocks, int64_t blockLength, int64_t stride)
      : packer_(0, blockLength, numBlocks, stride, numBlocks * stride) {
    CUDA_RUNTIME(cudaMalloc(&src, numBlocks * stride));
    CUDA_RUNTIME(cudaMalloc(&dst, numBlocks * blockLength));
  }

  ~DevicePack2D() {
    CUDA_RUNTIME(cudaFree(src));
    CUDA_RUNTIME(cudaFree(dst));
  }

  Benchmark::Sample run_iter() override {
    int pos = 0;
    Sample res{};
    double start = MPI_Wtime();
    packer_.pack(dst, &pos, src, 1);
    res.time = MPI_Wtime() - start;
    return res;
  }
};

class DeviceUnpack2D : public Benchmark {
  char *src, *dst;
  int64_t numBlocks_;
  int64_t blockLength_;
  int64_t stride_;
  Packer2D packer_;

public:
  DeviceUnpack2D(int64_t numBlocks, int64_t blockLength, int64_t stride)
      : packer_(0, blockLength, numBlocks, stride, numBlocks * stride) {
    CUDA_RUNTIME(cudaMalloc(&src, numBlocks * blockLength));
    CUDA_RUNTIME(cudaMalloc(&dst, numBlocks * stride));
  }

  ~DeviceUnpack2D() {
    CUDA_RUNTIME(cudaFree(src));
    CUDA_RUNTIME(cudaFree(dst));
  }

  Benchmark::Sample run_iter() override {
    int pos = 0;
    Sample res{};
    double start = MPI_Wtime();
    packer_.unpack(src, &pos, dst, 1);
    res.time = MPI_Wtime() - start;
    return res;
  }
};

/* Synchronous pack, as used in MPI_Send
 */
class HostPack2D : public Benchmark {
  char *src, *dst;
  int64_t numBlocks_;
  int64_t blockLength_;
  int64_t stride_;
  Packer2D packer_;

public:
  HostPack2D(int64_t numBlocks, int64_t blockLength, int64_t stride)
      : numBlocks_(numBlocks), blockLength_(blockLength),
        packer_(0, blockLength, numBlocks, stride, numBlocks * stride) {
    CUDA_RUNTIME(cudaMalloc(&src, numBlocks * stride));
    dst = hostAllocator.allocate(numBlocks * blockLength);
  }

  ~HostPack2D() {
    CUDA_RUNTIME(cudaFree(src));
    hostAllocator.deallocate(dst, numBlocks_ * blockLength_);
  }

  Benchmark::Sample run_iter() override {
    int pos = 0;
    Sample res{};
    double start = MPI_Wtime();
    packer_.pack(dst, &pos, src, 1);
    res.time = MPI_Wtime() - start;
    return res;
  }
};

class HostUnpack2D : public Benchmark {
  char *src, *dst;
  int64_t numBlocks_;
  int64_t blockLength_;
  int64_t stride_;
  Packer2D packer_;

public:
  HostUnpack2D(int64_t numBlocks, int64_t blockLength, int64_t stride)
      : numBlocks_(numBlocks), blockLength_(blockLength),
        packer_(0, blockLength, numBlocks, stride, numBlocks * stride) {
    src = hostAllocator.allocate(numBlocks * blockLength);
    CUDA_RUNTIME(cudaMalloc(&dst, numBlocks * stride));
  }

  ~HostUnpack2D() {
    hostAllocator.deallocate(src, numBlocks_ * blockLength_);
    src = {};
    CUDA_RUNTIME(cudaFree(dst));
  }

  Benchmark::Sample run_iter() override {
    int pos = 0;
    Sample res{};
    double start = MPI_Wtime();
    packer_.unpack(src, &pos, dst, 1);
    res.time = MPI_Wtime() - start;
    return res;
  }
};

/* fill any missing entries in sp
 */
void measure_system_performance(SystemPerformance &sp, MPI_Comm comm) {

  using topology::node_of_rank;

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "CUDA kernel\n";
    std::cerr << "bytes,s,niters\n";
  }
  if (rank == 0 && sp.cudaKernelLaunch == 0) {
    KernelLaunchBenchmark bm;
    Benchmark::Result res = bm.run(Benchmark::RunConfig());
    std::cerr << "=== " << res.trimean << " " << res.nIters << " ===\n";
    sp.cudaKernelLaunch = res.trimean;
  }

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "D2H\n";
    std::cerr << "bytes,s,niters\n";
  }
  if (rank == 0 && sp.d2h.empty()) {
    for (int i = 1; i <= 8 * 1024 * 1024; i *= 2) {
      CudaMemcpyAsyncD2H bm(i);
      Benchmark::Result res = bm.run(Benchmark::RunConfig());
      if (0 == rank) {
        std::cerr << i << "," << res.trimean << "," << res.nIters << "\n";
      }
      sp.d2h.push_back(IidTime{.time = res.trimean, .iid = res.iid});
    }
  }

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "H2D\n";
    std::cerr << "bytes,s,niters\n";
  }
  if (rank == 0 && sp.h2d.empty()) {
    for (int i = 1; i <= 8 * 1024 * 1024; i *= 2) {
      CudaMemcpyAsyncH2D bm(i);
      Benchmark::Result res = bm.run(Benchmark::RunConfig());
      if (0 == rank) {
        std::cerr << i << "," << res.trimean << "," << res.nIters << "\n";
      }
      sp.h2d.push_back(IidTime{.time = res.trimean, .iid = res.iid});
    }
  }

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "intra-node CPU-CPU\n";
    std::cerr << "bytes,s\n";
  }
  if (size >= 2 && (node_of_rank(comm, 0) == node_of_rank(comm, 1)) &&
      sp.intraNodeCpuCpuPingpong.empty()) {
    for (int i = 1; i <= 8 * 1024 * 1024; i *= 2) {
      CpuCpuPingpong bm(i, MPI_COMM_WORLD);
      Benchmark::Result res = bm.run(Benchmark::RunConfig());
      if (0 == rank) {
        std::cerr << i << "," << res.trimean << "\n";
      }

      sp.intraNodeCpuCpuPingpong.push_back(
          IidTime{.time = res.trimean, .iid = res.iid});
    }
  }

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "intra-node GPU-GPU\n";
    std::cerr << "bytes,s\n";
  }
  if (size >= 2 && (node_of_rank(comm, 0) == node_of_rank(comm, 1)) &&
      sp.intraNodeGpuGpuPingpong.empty()) {

    for (int i = 1; i <= 8 * 1024 * 1024; i *= 2) {
      GpuGpuPingpong bm(i, MPI_COMM_WORLD);
      Benchmark::Result res = bm.run(Benchmark::RunConfig());
      if (0 == rank) {
        std::cerr << i << "," << res.trimean << "\n";
      }

      sp.intraNodeGpuGpuPingpong.push_back(
          IidTime{.time = res.trimean, .iid = res.iid});
    }
  }

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "inter-node CPU-CPU\n";
    std::cerr << "bytes,s\n";
  }
  if (size >= 2 && (node_of_rank(comm, 0) != node_of_rank(comm, 1)) &&
      sp.interNodeCpuCpuPingpong.empty()) {
    for (int i = 1; i <= 8 * 1024 * 1024; i *= 2) {
      CpuCpuPingpong bm(i, MPI_COMM_WORLD);
      Benchmark::Result res = bm.run(Benchmark::RunConfig());
      if (0 == rank) {
        std::cerr << i << "," << res.trimean << "\n";
      }

      sp.interNodeCpuCpuPingpong.push_back(
          IidTime{.time = res.trimean, .iid = res.iid});
    }
  } else {
    LOG_WARN("skip interNodeCpuCpuPingpong");
  }

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "inter-node GPU-GPU\n";
    std::cerr << "bytes,s\n";
  }
  if (size >= 2 && (node_of_rank(comm, 0) != node_of_rank(comm, 1)) &&
      sp.interNodeGpuGpuPingpong.empty()) {
    for (int i = 1; i <= 8 * 1024 * 1024; i *= 2) {
      GpuGpuPingpong bm(i, MPI_COMM_WORLD);
      Benchmark::Result res = bm.run(Benchmark::RunConfig());
      if (0 == rank) {
        std::cerr << i << "," << res.trimean << "\n";
      }

      sp.interNodeGpuGpuPingpong.push_back(
          IidTime{.time = res.trimean, .iid = res.iid});
    }
  } else {
    LOG_WARN("skip interNodeGpuGpuPingpong");
  }

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "HostPack2D\n";
    std::cerr << "bytes,blockLength,s,niters\n";
  }
  if (rank == 0 && sp.packHost.empty()) {
    for (int i = 0; i < 9; ++i) {
      sp.packHost.push_back({});
      for (int j = 0; j < 9; ++j) {
        int64_t bytes = 1ull << (2 * i + 6);
        int64_t blockLength = 1ull << j;
        // no blocklength larger than bytes
        blockLength = min(bytes, blockLength);
        int64_t numBlocks = bytes / blockLength;
        HostPack2D bm(numBlocks, blockLength, 512);
        Benchmark::Result res = bm.run(Benchmark::RunConfig());
        std::cerr << bytes << "," << blockLength << "," << res.trimean << ","
                  << res.nIters << "\n";
        sp.packHost[i].push_back(IidTime{.time = res.trimean, .iid = res.iid});
      }
    }
  }

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "HostUnpack2D\n";
    std::cerr << "bytes,blockLength,s,niters\n";
  }
  if (rank == 0 && sp.unpackHost.empty()) {
    for (int i = 0; i < 9; ++i) {
      sp.unpackHost.push_back({});
      for (int j = 0; j < 9; ++j) {
        int64_t bytes = 1ull << (2 * i + 6);
        int64_t blockLength = 1ull << j;
        // no blocklength larger than bytes
        blockLength = min(bytes, blockLength);
        int64_t numBlocks = bytes / blockLength;
        HostUnpack2D bm(numBlocks, blockLength, 512);
        Benchmark::Result res = bm.run(Benchmark::RunConfig());
        std::cerr << bytes << "," << blockLength << "," << res.trimean << ","
                  << res.nIters << "\n";
        sp.unpackHost[i].push_back(
            IidTime{.time = res.trimean, .iid = res.iid});
      }
    }
  }

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "DevicePack2D\n";
    std::cerr << "bytes,blockLength,s,niters\n";
  }
  if (rank == 0 && sp.packDevice.empty()) {
    for (int i = 0; i < 9; ++i) {
      sp.packDevice.push_back({});
      for (int j = 0; j < 9; ++j) {
        int64_t bytes = 1ull << (2 * i + 6);
        int64_t blockLength = 1ull << j;
        // no blocklength larger than bytes
        blockLength = min(bytes, blockLength);
        int64_t numBlocks = bytes / blockLength;
        DevicePack2D bm(numBlocks, blockLength, 512);
        Benchmark::Result res = bm.run(Benchmark::RunConfig());
        std::cerr << bytes << "," << blockLength << "," << res.trimean << ","
                  << res.nIters << "\n";
        sp.packDevice[i].push_back(
            IidTime{.time = res.trimean, .iid = res.iid});
        ;
      }
    }
  }

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "DeviceUnpack2D\n";
    std::cerr << "bytes,blockLength,s,niters\n";
  }
  if (rank == 0 && sp.unpackDevice.empty()) {
    for (int i = 0; i < 9; ++i) {
      sp.unpackDevice.push_back({});
      for (int j = 0; j < 9; ++j) {
        int64_t bytes = 1ull << (2 * i + 6);
        int64_t blockLength = 1ull << j;
        // no blocklength larger than bytes
        blockLength = min(bytes, blockLength);
        int64_t numBlocks = bytes / blockLength;
        DeviceUnpack2D bm(numBlocks, blockLength, 512);
        Benchmark::Result res = bm.run(Benchmark::RunConfig());
        std::cerr << bytes << "," << blockLength << "," << res.trimean << ","
                  << res.nIters << "\n";
        sp.unpackDevice[i].push_back(
            IidTime{.time = res.trimean, .iid = res.iid});
        ;
      }
    }
  }
}
