#include "measure_system.hpp"

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

double interp_time(const std::vector<IidTime> a, int64_t bytes) {

  if (a.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  uint8_t lb = log2_floor(bytes);
  uint8_t ub = log2_ceil(bytes);

  // too large, just scale up the largest time
  if (ub >= a.size()) {
    return a.back().time * bytes / (1ull << (a.size() - 1));
  } else if (lb == ub) {
    return a[lb].time;
  } else { // interpolate between points
    float num = bytes - (1ull << lb);
    float den = (1ull << ub) - (1ull << lb);
    float sf = num / den;
    return a[lb].time * (1 - sf) + a[ub].time * sf;
  }
}

double interp_2d(const std::vector<std::vector<IidTime>> a, int64_t bytes,
                 int64_t stride) {
  assert(stride <= 512);

  /*find the surrounding points for which we have measurements,
    as well as indices into the measurement array for the points

    x1,y1 is lower corner, x2,y2 is higher corner

    the x coverage is complete,
     so we only have to handle the case when y1/y2 is not in the array
  */
  uint8_t yi1 = (log2_floor(bytes) - 6) / 2;
  int64_t y1 = 1ull << (yi1 * 2 + 6);
  uint8_t yi2 = (y1 == bytes) ? yi1 : yi1 + 1;
  int64_t y2 = 1ull << (yi2 * 2 + 6);

  // std::cerr << "yi1,yi2=" << int(yi1) << "," << int(yi2) << "\n";

  uint8_t xi1 = log2_floor(stride);
  int64_t x1 = 1ull << xi1;
  uint8_t xi2 = log2_ceil(stride);
  int64_t x2 = 1ull << xi2;

  int64_t x = stride;
  int64_t y = bytes;

  float sf_x;
  if (xi2 == xi1) {
    sf_x = 0.0;
  } else {
    sf_x = float(x - x1) / float(x2 - x1);
  }
  float sf_y;
  if (yi2 == yi1) {
    sf_y = 0.0;
  } else {
    sf_y = float(y - y1) / float(y2 - y1);
  }

  // message is too big, just scale the stride interpolation
  if (yi2 >= a.size()) {
    float base = (1 - sf_x) * a[a.size() - 1][xi1].time +
                 sf_x * a[a.size() - 1][xi2].time;
    float y_max = 1ull << ((a.size() - 1) * 2 + 6);
    // std::cerr << base << "," << y_max << " " << bytes << "\n";
    return base / y_max * bytes;
  } else {
    float f_x_y1 = (1 - sf_x) * a[yi1][xi1].time + sf_x * a[yi1][xi2].time;
    float f_x_y2 = (1 - sf_x) * a[yi2][xi1].time + sf_x * a[yi2][xi2].time;
    float f_x_y = (1 - sf_y) * f_x_y1 + sf_y * f_x_y2;
    return f_x_y;
  }
}

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

public:
  // zero buffer to put it in cache
  CpuCpuPingpong(size_t n, MPI_Comm comm) : buf_(n), MpiBenchmark(comm) {}
  ~CpuCpuPingpong() {}

  Benchmark::Sample run_iter() override {
    int rank;
    MPI_Comm_rank(comm_, &rank);
    MPI_Barrier(comm_);
    Time start = Clock::now();
    for (int i = 0; i < nreps_; ++i) {
      if (0 == rank) {
        libmpi.MPI_Send(buf_.data(), buf_.size(), MPI_BYTE, 1, 0, comm_);
        libmpi.MPI_Recv(buf_.data(), buf_.size(), MPI_BYTE, 1, 0, comm_,
                        MPI_STATUS_IGNORE);
      } else if (1 == rank) {
        libmpi.MPI_Recv(buf_.data(), buf_.size(), MPI_BYTE, 0, 0, comm_,
                        MPI_STATUS_IGNORE);
        libmpi.MPI_Send(buf_.data(), buf_.size(), MPI_BYTE, 0, 0, comm_);
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

class DevicePack2D : public Benchmark {
  cudaStream_t stream;
  cudaEvent_t start, stop;
  char *src, *dst;
  int64_t numBlocks_;
  int64_t blockLength_;
  int64_t stride_;
  Packer2D packer_;

public:
  DevicePack2D(int64_t numBlocks, int64_t blockLength, int64_t stride)
      : packer_(0, blockLength, numBlocks, stride) {
    CUDA_RUNTIME(cudaStreamCreate(&stream));
    CUDA_RUNTIME(cudaEventCreate(&start));
    CUDA_RUNTIME(cudaEventCreate(&stop));
    CUDA_RUNTIME(cudaMalloc(&src, numBlocks * stride));
    CUDA_RUNTIME(cudaMalloc(&dst, numBlocks * stride));
  }

  ~DevicePack2D() {
    CUDA_RUNTIME(cudaStreamDestroy(stream));
    CUDA_RUNTIME(cudaEventDestroy(start));
    CUDA_RUNTIME(cudaEventDestroy(stop));
    CUDA_RUNTIME(cudaFree(src));
    CUDA_RUNTIME(cudaFree(dst));
  }

  Benchmark::Sample run_iter() override {
    float ms;
    int pos = 0;
    packer_.launch_pack(dst, &pos, src, 1, stream, start, stop);
    CUDA_RUNTIME(cudaEventSynchronize(stop));
    CUDA_RUNTIME(cudaEventElapsedTime(&ms, start, stop));
    Sample res{};
    res.time = ms / 1024.0;
    return res;
  }
};

class PackHost2D : public Benchmark {
  cudaStream_t stream;
  cudaEvent_t start, stop;
  char *src, *dst;
  int64_t numBlocks_;
  int64_t blockLength_;
  int64_t stride_;
  Packer2D packer_;

public:
  PackHost2D(int64_t numBlocks, int64_t blockLength, int64_t stride)
      : packer_(0, blockLength, numBlocks, stride) {
    CUDA_RUNTIME(cudaStreamCreate(&stream));
    CUDA_RUNTIME(cudaEventCreate(&start));
    CUDA_RUNTIME(cudaEventCreate(&stop));
    CUDA_RUNTIME(cudaMalloc(&src, numBlocks * stride));
    dst = new char[numBlocks * stride];
    CUDA_RUNTIME(
        cudaHostRegister(dst, numBlocks * stride, cudaHostRegisterPortable));
  }

  ~PackHost2D() {
    CUDA_RUNTIME(cudaStreamDestroy(stream));
    CUDA_RUNTIME(cudaEventDestroy(start));
    CUDA_RUNTIME(cudaEventDestroy(stop));
    CUDA_RUNTIME(cudaFree(src));
    CUDA_RUNTIME(cudaHostUnregister(dst));
    delete[] dst;
  }

  Benchmark::Sample run_iter() override {
    float ms;
    int pos = 0;
    packer_.launch_pack(dst, &pos, src, 1, stream, start, stop);
    CUDA_RUNTIME(cudaEventSynchronize(stop));
    CUDA_RUNTIME(cudaEventElapsedTime(&ms, start, stop));
    Sample res{};
    res.time = ms / 1024.0;
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
    std::cerr << "PackHost2D\n";
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
        PackHost2D bm(numBlocks, blockLength, 512);
        Benchmark::Result res = bm.run(Benchmark::RunConfig());
        std::cerr << bytes << "," << blockLength << "," << res.trimean << ","
                  << res.nIters << "\n";
        sp.packHost[i].push_back(IidTime{.time = res.trimean, .iid = res.iid});
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
}
