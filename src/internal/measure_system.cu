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

/* extern*/ KernelLaunch kernelLaunch;

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

class CpuCpuPingpong : public MpiBenchmark {
  std::vector<char> buf_;

public:
  CpuCpuPingpong(size_t n, MPI_Comm comm) : buf_(n), MpiBenchmark(comm) {}
  ~CpuCpuPingpong() {}

  Benchmark::IterResult run_iter() override {

    int rank;
    MPI_Comm_rank(comm_, &rank);
    int reps = 2000 / buf_.size();
    if (reps < 10)
      reps = 10;
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

bool benchmark_system(MPI_Comm comm) {

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (rank == 0) {
    KernelLaunchBenchmark bm;
    Benchmark::Result res = bm.run();
    std::cerr << "=== " << res.trimean << " " << res.nIters << " ===\n";
    kernelLaunch.secs = res.trimean;
  }

  MPI_Barrier(comm);

  if (size >= 2) {
    for (int i = 1; i < 1024 * 1024; i *= 2) {
      CpuCpuPingpong bm(i, MPI_COMM_WORLD);
      Benchmark::Result res = bm.run();
      std::cerr << "=== " << i << "B " << res.trimean << " " << res.nIters
                << " ===\n";
      kernelLaunch.secs = res.trimean;
    }
  }

  return true;
}

bool save_benchmark_cache(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (0 == rank) {
    LOG_ERROR("save_benchmark_cache unimplemented");
  }
  return false;
}

void measure_system(MPI_Comm comm) {

  if (!load_benchmark_cache(comm)) {
    benchmark_system(comm);
    save_benchmark_cache(comm);
  }
}
