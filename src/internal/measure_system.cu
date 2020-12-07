#include "benchmark.hpp"
#include "cuda_runtime.hpp"
#include "measure_system.hpp"

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

void measure_system(MPI_Comm comm) {

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  if (rank == 0) {
    Benchmark *bm = new KernelLaunchBenchmark();
    Benchmark::Result res = bm->run();
    std::cerr << "=== " << res.trimean << " " << res.nIters << " ===\n";
    delete bm;
    kernelLaunch.secs = res.trimean;
  }
}
