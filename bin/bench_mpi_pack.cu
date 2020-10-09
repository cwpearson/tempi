#include "../include/cuda_runtime.hpp"
#include "../include/env.hpp"
#include "../test/support/type.hpp"
#include "statistics.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <chrono>
#include <sstream>

typedef std::chrono::system_clock Clock;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::time_point<Clock, Duration> Time;

struct BenchResult {
  double packTime;
};

BenchResult bench(MPI_Datatype ty, const Dim3 ext,
                  bool tempi, // use tempi or not
                  const int nIters) {

  // configure TEMPI
  environment::noTypeCommit = !tempi;
  environment::noPack = !tempi;

  // allocation extent (B)
  cudaExtent allocExt = {};
  allocExt.width = 1024;
  allocExt.height = 1024;
  allocExt.depth = 1024;

  // create device allocations
#ifdef HOST
  char *src = new char[allocExt.width * allocExt.height * allocExt.depth];
#else
  cudaPitchedPtr src = {};
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc3D(&src, allocExt));
  allocExt.width = src.pitch; // cudaMalloc3D may adjust pitch
#endif

  // copy extent (B)
  cudaExtent copyExt = {};
  copyExt.width = ext.x;
  copyExt.height = ext.y;
  copyExt.depth = ext.z;

  // create flat destination allocation
  char *dst = nullptr;
  const int dstSize = copyExt.width * copyExt.height * copyExt.depth;
#ifdef HOST
  dst = new char[dstSize];
#else
  CUDA_RUNTIME(cudaMalloc(&dst, dstSize));
#endif

  Statistics stats;
  nvtxRangePush("loop");
  for (int n = 0; n < nIters; ++n) {
    int position = 0;
    auto start = Clock::now();
#ifdef HOST
    MPI_Pack(src, 1, ty, dst, dstSize, &position, MPI_COMM_WORLD);
#else
    MPI_Pack(src.ptr, 1, ty, dst, dstSize, &position, MPI_COMM_WORLD);
#endif
    auto stop = Clock::now();
    Duration dur = stop - start;
    stats.insert(dur.count());
  }
  nvtxRangePop();

#ifdef HOST
  delete[] src;
  delete[] dst;
#else
  CUDA_RUNTIME(cudaFree(src.ptr));
  CUDA_RUNTIME(cudaFree(dst));
#endif

  return BenchResult{.packTime = stats.trimean()};
}

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  int nIters = 30;

  Dim3 allocExt(1024, 1024, 1024);
  BenchResult result;



  std::vector<Dim3> dims = {
      Dim3(1, 1024, 1024), Dim3(2, 1024, 512),  Dim3(4, 1024, 256),
      Dim3(8, 1024, 128),  Dim3(16, 1024, 64),  Dim3(32, 1024, 32),
      Dim3(64, 1024, 16),  Dim3(128, 1024, 8),  Dim3(256, 1024, 4),
      Dim3(512, 1024, 2),  Dim3(1024, 1024, 1), Dim3(1, 1024, 1),
      Dim3(2, 1024, 1),    Dim3(4, 1024, 1),    Dim3(8, 1024, 1),
      Dim3(16, 1024, 1),   Dim3(32, 1024, 1),   Dim3(64, 1024, 1),
      Dim3(128, 1024, 1),  Dim3(256, 1024, 1),  Dim3(512, 1024, 1),
      Dim3(12, 512, 512),  Dim3(512, 3, 512),   Dim3(512, 512, 3)};
  std::vector<bool> tempis = {true};

  std::cout << "s,x,y,z,hib (MiB/s),v_hv_hv (MiB/s),v_hv (MiB/s)\n";

  for (Dim3 ext : dims) {

    for (bool tempi : tempis) {

      std::string s;
      s = std::to_string(ext.x) + "/" + std::to_string(ext.y) + "/" +
          std::to_string(ext.z) + "/" + std::to_string(tempi);

      std::cout << s;
      std::cout << "," << ext.x << "," << ext.y << "," << ext.z;
      std::cout << "," << tempi;
      std::cout << std::flush;

      MPI_Datatype ty;

#if 0
    ty = make_hi(ext, allocExt);
    MPI_Type_commit(&ty);
    result = bench(ty, ext, tempi, nIters);
    std::cout << "," << double(ext.flatten()) / 1024 / 1024 / result.packTime;
    std::cout << std::flush;
#endif

#if 0
    ty = make_hib(ext, allocExt);
    MPI_Type_commit(&ty);
    result = bench(ty, ext, tempi, nIters);
    std::cout << "," << double(ext.flatten()) / 1024 / 1024 / result.packTime;
    std::cout << std::flush;
#endif

#if 1
    ty = make_v1_hv_hv(ext, allocExt);
    MPI_Type_commit(&ty);
    result = bench(ty, ext, tempi, nIters);
    std::cout << "," << double(ext.flatten()) / 1024 / 1024 / result.packTime;
    std::cout << std::flush;
#endif

#if 1
    ty = make_v_hv(ext, allocExt);
    MPI_Type_commit(&ty);
    result = bench(ty, ext, tempi, nIters);
    std::cout << "," << double(ext.flatten()) / 1024 / 1024 / result.packTime;
    std::cout << std::flush;
#endif

    std::cout << "\n";
    std::cout << std::flush;
    nvtxRangePop();
    }
  }

  MPI_Finalize();

  return 0;
}