#include "benchmark.hpp"
#include "cuda_runtime.hpp"
#include "logging.hpp"
#include "measure_system.hpp"
#include "symbols.hpp"

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

  Benchmark::IterResult run_iter() override {
    IterResult res{};

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
    dst = new char[n];
    CUDA_RUNTIME(cudaHostRegister(dst, n, cudaHostRegisterPortable));
  }

  ~CudaMemcpyAsyncD2H() {
    CUDA_RUNTIME(cudaStreamDestroy(stream));
    CUDA_RUNTIME(cudaFree(src));
    CUDA_RUNTIME(cudaHostUnregister(dst));
    delete[] dst;
  }

  Benchmark::IterResult run_iter() override {
    IterResult res{};
    Time start = Clock::now();
    const int nreps = 10;
    for (int i = 0; i < nreps; ++i) {
      cudaMemcpyAsync(dst, src, n_, cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
    }
    Time stop = Clock::now();
    CUDA_RUNTIME(cudaGetLastError());
    Duration dur = stop - start;
    res.time = dur.count() / double(nreps);
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
  }

  ~CudaMemcpyAsyncH2D() {
    CUDA_RUNTIME(cudaStreamDestroy(stream));
    CUDA_RUNTIME(cudaFree(dst));
    CUDA_RUNTIME(cudaHostUnregister(src));
    delete[] src;
  }

  Benchmark::IterResult run_iter() override {
    IterResult res{};
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

/* a cpu-to-cpu MPI ping-pong test
 */
class CpuCpuPingpong : public MpiBenchmark {
  std::vector<char> buf_;

public:
  CpuCpuPingpong(size_t n, MPI_Comm comm) : buf_(n), MpiBenchmark(comm) {}
  ~CpuCpuPingpong() {}

  Benchmark::IterResult run_iter() override {

    int rank;
    MPI_Comm_rank(comm_, &rank);
    int reps = 1000 / buf_.size();
    if (reps < 2)
      reps = 2;
    Time start = Clock::now();
    for (int i = 0; i < reps; ++i) {

      MPI_Barrier(comm_);
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

    IterResult res{};
    res.time = maxTime / double(reps);
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

SystemPerformance measure_system_performance(MPI_Comm comm) {

  SystemPerformance sp{};

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (rank == 0) {
    KernelLaunchBenchmark bm;
    Benchmark::Result res = bm.run();
    std::cerr << "=== " << res.trimean << " " << res.nIters << " ===\n";
    sp.cudaKernelLaunch = res.trimean;
  }
  return sp;

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "D2H\n";
    std::cerr << "bytes,s,niters\n";
  }
  if (rank == 0) {
    for (int i = 1; i < 1024 * 1024; i *= 2) {
      CudaMemcpyAsyncD2H bm(i);
      Benchmark::Result res = bm.run();
      if (0 == rank) {
        std::cerr << i << "," << res.trimean << "," << res.nIters << "\n";
      }
      sp.d2h.push_back(Bandwidth{.bytes = i, .time = res.trimean, .iid=res.iid});
    }
  }

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "H2D\n";
    std::cerr << "bytes,s,niters\n";
  }
  if (rank == 0) {
    for (int i = 1; i < 1024 * 1024; i *= 2) {
      CudaMemcpyAsyncH2D bm(i);
      Benchmark::Result res = bm.run();
      if (0 == rank) {
        std::cerr << i << "," << res.trimean << "," << res.nIters << "\n";
      }
      sp.h2d.push_back(Bandwidth{.bytes = i, .time = res.trimean, .iid=res.iid});
    }
  }

  MPI_Barrier(comm);
  if (0 == rank) {
    std::cerr << "bytes,s\n";
  }
  if (size >= 2) {
    for (int i = 1; i < 1024 * 1024; i *= 2) {
      CpuCpuPingpong bm(i, MPI_COMM_WORLD);
      Benchmark::Result res = bm.run();
      if (0 == rank) {
        std::cerr << i << "," << res.trimean << "\n";
      }

      sp.pingpong.push_back(Bandwidth{.bytes = i, .time = res.trimean, .iid=res.iid});
    }
  }

  return sp;
}
