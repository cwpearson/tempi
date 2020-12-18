#include "../include/cuda_runtime.hpp"
#include "../include/env.hpp"
#include "../include/type_cache.hpp"
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
                  const char *name = "<unnamed>") {

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Type_commit(&ty);
  int count = 1;

  MPI_Aint lb, typeExtent;
  MPI_Type_get_extent(ty, &lb, &typeExtent);
  int typeSize;
  MPI_Type_size(ty, &typeSize);

  char *buf = {};
  CUDA_RUNTIME(cudaMalloc(&buf, typeExtent * count));

  Statistics packStats, unpackStats;
  nvtxRangePush(name);
  for (int n = 0; n < nIters; ++n) {
    int pos = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (0 == rank) {
      MPI_Send(buf, count, ty, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(buf, count, ty, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      MPI_Recv(buf, count, ty, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send(buf, count, ty, 0, 0, MPI_COMM_WORLD);
    }
    double stop = MPI_Wtime();
    packStats.insert(stop - start);
    unpackStats.insert(stop - start);
  }
  nvtxRangePop();

  CUDA_RUNTIME(cudaFree(buf));

  return BenchResult{.size = typeSize * count,
                     .packTime = packStats.trimean(),
                     .unpackTime = unpackStats.trimean()};
}

struct Factory2D {
  TypeFactory2D fn;
  const char *name;
};

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (2 != size) {
    std::cerr << "only 2 rank plz\n";
    MPI_Finalize();
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

  int stride = 512;
  bool stage;

  if (0 == rank) {
    for (int contig : contigs) {
      std::cout << "," << contig;
    }
    std::cout << std::endl << std::flush;
  }

  for (int target : targets) {
    if (0 == rank) {
      std::cout << target;
    }
    for (int contig : contigs) {
      if (contig > target) {
        contig = target;
      }

      int numBlocks = target / contig;

      std::string s;
      s += std::to_string(target);
      s += "|" + std::to_string(contig);
      MPI_Datatype ty = make_2d_byte_vector(numBlocks, contig, stride);
      result = bench(ty, nIters, s.c_str());
      MPI_Type_free(&ty);

      if (0 == rank) {
        std::cout << ",";
        std::cout << result.packTime;
        std::cout << std::flush;
      }
    }
    if (0 == rank) {
      std::cout << std::endl << std::flush;
    }
  }

  MPI_Finalize();
  return 0;
}
