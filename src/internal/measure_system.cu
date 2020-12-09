#include "benchmark.hpp"
#include "cuda_runtime.hpp"
#include "logging.hpp"
#include "measure_system.hpp"
#include "symbols.hpp"
#include "topology.hpp"

#include <mpi.h>

#include <chrono>
#include <iostream>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::time_point<Clock, Duration> Time;

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

class CudaMemcpyAsyncD2H : public Benchmark {
  cudaStream_t stream;
  char *src, *dst;
  size_t n_;
  int nreps_;

public:
  CudaMemcpyAsyncD2H(size_t n) : n_(n) {
    CUDA_RUNTIME(cudaStreamCreate(&stream));
    CUDA_RUNTIME(cudaMalloc(&src, n));
    dst = new char[n];
    CUDA_RUNTIME(cudaHostRegister(dst, n, cudaHostRegisterPortable));
  }

  ~CudaMemcpyAsyncD2H() {
    CUDA_RUNTIME(cudaStreamDestroy(stream));
    CUDA_RUNTIME(cudaFree(src));
    CUDA_RUNTIME(cudaHostUnregister(dst));
    delete[] dst;
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
    src = new char[n];
    CUDA_RUNTIME(cudaHostRegister(src, n, cudaHostRegisterPortable));
    // std::memset(src, -1, n); // cache host buffer
  }

  ~CudaMemcpyAsyncH2D() {
    CUDA_RUNTIME(cudaStreamDestroy(stream));
    CUDA_RUNTIME(cudaFree(dst));
    CUDA_RUNTIME(cudaHostUnregister(src));
    delete[] src;
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
 */
class CpuCpuPingpong : public MpiBenchmark {
  std::vector<char> buf_;
  int nreps_;

public:
  // zero buffer to put it in cache
  CpuCpuPingpong(size_t n, MPI_Comm comm) : buf_(n), MpiBenchmark(comm) {}
  ~CpuCpuPingpong() {}

  void setup() {
    int rank;
    MPI_Comm_rank(comm_, &rank);
    nreps_ = 1;
    for (int i = 0; i < 4; ++i) {
      Benchmark::Sample s = run_iter(); // measure time
      nreps_ = 200e-6 / s.time;
      nreps_ = std::max(nreps_, 1);
    }
    if (0 == rank) {
      LOG_DEBUG("estimate nreps_=" << nreps_ << " for 200us");
    }
    MPI_Bcast(&nreps_, 1, MPI_INT, 0, comm_);
  }

  Benchmark::Sample run_iter() override {

    int rank;
    MPI_Comm_rank(comm_, &rank);
    MPI_Barrier(comm_);
    Time start = Clock::now();
    for (int i = 0; i < nreps_; ++i) {
      if (0 == rank) {
        libmpi.MPI_Send(buf_.data(), buf_.size(), MPI_BYTE, 1, 0, comm_);
      } else if (1 == rank) {
        libmpi.MPI_Recv(buf_.data(), buf_.size(), MPI_BYTE, 0, 0, comm_,
                        MPI_STATUS_IGNORE);
      }
      if (0 == rank) {
        libmpi.MPI_Recv(buf_.data(), buf_.size(), MPI_BYTE, 1, 0, comm_,
                        MPI_STATUS_IGNORE);
      } else if (1 == rank) {
        libmpi.MPI_Send(buf_.data(), buf_.size(), MPI_BYTE, 0, 0, comm_);
      }
    }
    Time stop = Clock::now();
    Duration dur = stop - start;

    double time = dur.count(), maxTime;
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
  int nreps_;
  size_t n_;

  void setup() {
    int rank;
    MPI_Comm_rank(comm_, &rank);
    nreps_ = 1;
    for (int i = 0; i < 4; ++i) {
      Benchmark::Sample s = run_iter(); // measure time
      nreps_ = 200e-6 / s.time;
      nreps_ = std::max(nreps_, 1);
    }
    if (0 == rank) {
      LOG_DEBUG("estimate nreps_=" << nreps_ << " for 200us");
    }
    MPI_Bcast(&nreps_, 1, MPI_INT, 0, comm_);
  }

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
      } else if (1 == rank) {
        libmpi.MPI_Recv(buf_, n_, MPI_BYTE, 0, 0, comm_, MPI_STATUS_IGNORE);
      }
      if (0 == rank) {
        libmpi.MPI_Recv(buf_, n_, MPI_BYTE, 1, 0, comm_, MPI_STATUS_IGNORE);
      } else if (1 == rank) {
        libmpi.MPI_Send(buf_, n_, MPI_BYTE, 0, 0, comm_);
      }
    }
    Time stop = Clock::now();
    Duration dur = stop - start;

    double time = dur.count(), maxTime;
    MPI_Allreduce(&time, &maxTime, 1, MPI_DOUBLE, MPI_MAX, comm_);

    Sample res{};
    res.time = maxTime / double(nreps_);

    return res;
  }
};

bool load_benchmark_cache(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0) {
    LOG_ERROR("load_benchmark_cache unimpemented");
  }
  return false;
}

/* fill any missing entries in sp
 */
void measure_system_performance(SystemPerformance &sp, MPI_Comm comm) {

  using topology::node_of_rank;

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

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
    for (int i = 1; i < 1024 * 1024; i *= 2) {
      CudaMemcpyAsyncD2H bm(i);
      Benchmark::Result res = bm.run(Benchmark::RunConfig());
      if (0 == rank) {
        std::cerr << i << "," << res.trimean << "," << res.nIters << "\n";
      }
      sp.d2h.push_back(
          Bandwidth{.bytes = i, .time = res.trimean, .iid = res.iid});
    }
  }

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "H2D\n";
    std::cerr << "bytes,s,niters\n";
  }
  if (rank == 0 && sp.h2d.empty()) {
    for (int i = 1; i < 1024 * 1024; i *= 2) {
      CudaMemcpyAsyncH2D bm(i);
      Benchmark::Result res = bm.run(Benchmark::RunConfig());
      if (0 == rank) {
        std::cerr << i << "," << res.trimean << "," << res.nIters << "\n";
      }
      sp.h2d.push_back(
          Bandwidth{.bytes = i, .time = res.trimean, .iid = res.iid});
    }
  }

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "intra-node CPU-CPU\n";
    std::cerr << "bytes,s\n";
  }
  LOG_INFO(node_of_rank(comm, 0));
  LOG_INFO(node_of_rank(comm, 1));
  LOG_INFO("empty? " << sp.intraNodeCpuCpuPingpong.empty());
  if (size >= 2 && (node_of_rank(comm, 0) == node_of_rank(comm, 1)) &&
      sp.intraNodeCpuCpuPingpong.empty()) {
    for (int i = 1; i < 1024 * 1024; i *= 2) {
      CpuCpuPingpong bm(i, MPI_COMM_WORLD);
      Benchmark::Result res = bm.run(Benchmark::RunConfig());
      if (0 == rank) {
        std::cerr << i << "," << res.trimean << "\n";
      }

      sp.intraNodeCpuCpuPingpong.push_back(
          Bandwidth{.bytes = i, .time = res.trimean, .iid = res.iid});
    }
  }
  LOG_INFO("after intra-node CPU-CPU");

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "intra-node GPU-GPU\n";
    std::cerr << "bytes,s\n";
  }
  if (size >= 2 && (node_of_rank(comm, 0) == node_of_rank(comm, 1)) &&
      sp.intraNodeGpuGpuPingpong.empty()) {

    for (int i = 1; i < 1024 * 1024; i *= 2) {
      GpuGpuPingpong bm(i, MPI_COMM_WORLD);
      Benchmark::Result res = bm.run(Benchmark::RunConfig());
      if (0 == rank) {
        std::cerr << i << "," << res.trimean << "\n";
      }

      sp.intraNodeGpuGpuPingpong.push_back(
          Bandwidth{.bytes = i, .time = res.trimean, .iid = res.iid});
    }
  }

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "bytes,s\n";
  }
  if (size >= 2 && (node_of_rank(comm, 0) != node_of_rank(comm, 1)) &&
      sp.interNodeCpuCpuPingpong.empty()) {
    for (int i = 1; i < 1024 * 1024; i *= 2) {
      CpuCpuPingpong bm(i, MPI_COMM_WORLD);
      Benchmark::Result res = bm.run(Benchmark::RunConfig());
      if (0 == rank) {
        std::cerr << i << "," << res.trimean << "\n";
      }

      sp.interNodeCpuCpuPingpong.push_back(
          Bandwidth{.bytes = i, .time = res.trimean, .iid = res.iid});
    }
  } else {
    LOG_WARN("skip interNodeCpuCpuPingpong");
  }

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "bytes,s\n";
  }
  if (size >= 2 && (node_of_rank(comm, 0) != node_of_rank(comm, 1)) &&
      sp.interNodeGpuGpuPingpong.empty()) {
    for (int i = 1; i < 1024 * 1024; i *= 2) {
      GpuGpuPingpong bm(i, MPI_COMM_WORLD);
      Benchmark::Result res = bm.run(Benchmark::RunConfig());
      if (0 == rank) {
        std::cerr << i << "," << res.trimean << "\n";
      }

      sp.interNodeGpuGpuPingpong.push_back(
          Bandwidth{.bytes = i, .time = res.trimean, .iid = res.iid});
    }
  } else {
    LOG_WARN("skip interNodeGpuGpuPingpong");
  }
}
