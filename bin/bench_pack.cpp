#include "../include/cuda_runtime.hpp"
#include "../include/env.hpp"
#include "../include/packer_cache.hpp"
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
  int64_t size;
  double packTime;
  double unpackTime;
};

BenchResult bench(MPI_Datatype ty,  // message datatype
                  const int nIters, // iterations to measure
                  const bool stage, // pack into host / unpack from host
                  const char *name = "<unnamed>") {

  MPI_Type_commit(&ty);
  int count = 1;

  MPI_Aint lb, typeExtent;
  MPI_Type_get_extent(ty, &lb, &typeExtent);
  int typeSize;
  MPI_Type_size(ty, &typeSize);
  int packedSize;
  MPI_Pack_size(count, ty, MPI_COMM_WORLD, &packedSize);

  auto pi = packerCache.find(ty);
  if (packerCache.end() == pi) {
    return BenchResult{
        .size = typeSize * count, .packTime = 0, .unpackTime = 0};
  }

  char *src = {}, *dst = {};
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc(&src, typeExtent * count));
  if (stage) {
    CUDA_RUNTIME(cudaHostAlloc(&dst, packedSize, cudaHostAllocMapped));
  } else {
    CUDA_RUNTIME(cudaMalloc(&dst, packedSize));
  }

  Statistics packStats, unpackStats;
  nvtxRangePush(name);
  for (int n = 0; n < nIters; ++n) {
    int pos = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    pi->second->pack(dst, &pos, src, count);
    double stop = MPI_Wtime();
    packStats.insert(stop - start);

    pos = 0;
    start = MPI_Wtime();
    pi->second->unpack(dst, &pos, src, count);
    stop = MPI_Wtime();
    unpackStats.insert(stop - start);
  }
  nvtxRangePop();

  CUDA_RUNTIME(cudaFree(src));
  if (stage) {
    CUDA_RUNTIME(cudaFreeHost(dst));
  } else {
    CUDA_RUNTIME(cudaFree(dst));
  }

  return BenchResult{.size = typeSize * count,
                     .packTime = packStats.trimean(),
                     .unpackTime = unpackStats.trimean()};
}

struct Factory3D {
  TypeFactory3D fn;
  const char *name;
};

struct Factory2D {
  TypeFactory2D fn;
  const char *name;
};

struct Factory1D {
  TypeFactory1D fn;
  const char *name;
};

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  // only 1 rank
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (1 != size) {
    std::cerr << "only 1 rank plz\n";
    exit(-1);
  }

  int nIters;
  if (environment::noTempi) {
    nIters = 5;
  } else {
    nIters = 200;
  }

  std::vector<bool> stages{
      true}; // whether to one-shot pack device-host / unpack host-device

  BenchResult result;

  /* 2D packing
   */

  std::vector<int> targets{64,    256,    1024,    4096,       16384,
                           65536, 262144, 1048576, 1048576 * 4};
  std::vector<int> contigs{1, 2, 4, 8, 12, 16, 20, 24, 32, 64, 128, 256};

  int stride = 256;
  bool stage = true;

  for (int contig : contigs) {
    std::cout << "," << contig;
  }
  std::cout << std::endl << std::flush;

  for (int target : targets) {
    std::cout << target;
    for (int contig : contigs) {
      if (contig > target) {
        contig = target;
      }

      int numBlocks = target / contig;

      std::string s;
      s += std::to_string(target);
      s += "|" + std::to_string(contig);
      MPI_Datatype ty = make_2d_byte_vector(numBlocks, contig, stride);
      result = bench(ty, nIters, stage, s.c_str());
      MPI_Type_free(&ty);

      std::cout << ",";
      std::cout << result.packTime;
      std::cout << std::flush;
    }
    std::cout << std::endl << std::flush;
  }


for (int contig : contigs) {
    std::cout << "," << contig;
  }
  std::cout << std::endl << std::flush;

  for (int target : targets) {
    std::cout << target;
    for (int contig : contigs) {
      if (contig > target) {
        contig = target;
      }

      int numBlocks = target / contig;

      std::string s;
      s += std::to_string(target);
      s += "|" + std::to_string(contig);
      MPI_Datatype ty = make_2d_byte_vector(numBlocks, contig, stride);
      result = bench(ty, nIters, stage, s.c_str());
      MPI_Type_free(&ty);

      std::cout << ",";
      std::cout << result.unpackTime;
      std::cout << std::flush;
    }
    std::cout << std::endl << std::flush;
  }



  MPI_Finalize();
  return 0;
}
