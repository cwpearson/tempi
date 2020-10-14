#include "../include/cuda_runtime.hpp"
#include "../include/env.hpp"
#include "../support/type.hpp"
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
  environment::noTempi = !tempi;

  // allocation extent (B)
  cudaExtent allocExt = {};
  allocExt.width = 1024;
  allocExt.height = 1024;
  allocExt.depth = 1024;

  // create device allocations
  cudaPitchedPtr src = {};
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc3D(&src, allocExt));
  allocExt.width = src.pitch; // cudaMalloc3D may adjust pitch

  // copy extent (B)
  cudaExtent copyExt = {};
  copyExt.width = ext.x;
  copyExt.height = ext.y;
  copyExt.depth = ext.z;

  // create flat destination allocation
  char *dst = nullptr;
  const int dstSize = copyExt.width * copyExt.height * copyExt.depth;
  CUDA_RUNTIME(cudaMalloc(&dst, dstSize));

  Statistics stats;
  nvtxRangePush("loop");
  for (int n = 0; n < nIters; ++n) {
    int position = 0;
    auto start = Clock::now();
    MPI_Pack(src.ptr, 1, ty, dst, dstSize, &position, MPI_COMM_WORLD);
    auto stop = Clock::now();
    Duration dur = stop - start;
    stats.insert(dur.count());
  }
  nvtxRangePop();

  CUDA_RUNTIME(cudaFree(src.ptr));
  CUDA_RUNTIME(cudaFree(dst));

  return BenchResult{.packTime = stats.trimean()};
}

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  // only need to run on rank 0
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (0 != rank) {
    goto finalize;
  }

  { // prevent init bypass
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
  std::vector<bool> tempis = {true, false};

    std::cout << "s,x,y,z,hib (MiB/s),v_hv_hv (MiB/s),v_hv (MiB/s)\n";

  for (bool tempi : tempis) {
    for (Dim3 ext : dims) {

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
        std::cout << ","
                  << double(ext.flatten()) / 1024 / 1024 / result.packTime;
        std::cout << std::flush;
#endif

#if 1
        ty = make_v_hv(ext, allocExt);
        MPI_Type_commit(&ty);
        result = bench(ty, ext, tempi, nIters);
        std::cout << ","
                  << double(ext.flatten()) / 1024 / 1024 / result.packTime;
        std::cout << std::flush;
#endif

        std::cout << "\n";
        std::cout << std::flush;
        nvtxRangePop();
      }
    }
  }
finalize:
  MPI_Finalize();
  return 0;
}
