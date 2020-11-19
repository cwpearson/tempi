/*! \file

    Sends messages of a fixed size between ranks 0-N/2 and N/2-N
*/

#include "../include/cuda_runtime.hpp"
#include "../include/env.hpp"
#include "../include/logging.hpp"
#include "../support/type.hpp"
#include "statistics.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <sstream>

struct BenchResult {
  int64_t bytes;       // number of bytes sent by this rank
  double pingPongTime; // elapsed time
};

/* Each rank i and i + size/2 are paired for a send/recv
   reported result is the min of the max time seen across any rank
*/
BenchResult bench(MPI_Datatype ty,  // message datatype
                  int count,        // number of datatypes
                  bool host,        // host allocations (or device)
                  const int nIters, // iterations to measure
                  const char *name = "<unnamed>") {

  if (ty != MPI_BYTE) {
    MPI_Type_commit(&ty);
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Aint lb, extent;
  MPI_Type_get_extent(ty, &lb, &extent);
  int typeSize;
  MPI_Type_size(ty, &typeSize);

  // create input device allocations
  char *src = {}, *dst = {};
  if (host) {
    src = new char[extent * count];
    dst = new char[extent * count];
  } else {
    CUDA_RUNTIME(cudaSetDevice(0));
    CUDA_RUNTIME(cudaMalloc(&src, extent * count));
    CUDA_RUNTIME(cudaMalloc(&dst, extent * count));
  }

  Statistics stats;
  nvtxRangePush(name);
  for (int n = 0; n < nIters; ++n) {
    int position = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (rank < size / 2) {
      MPI_Send(src, count, ty, rank + size / 2, 0, MPI_COMM_WORLD);
      MPI_Recv(dst, count, ty, rank + size / 2, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

    } else {
      MPI_Recv(dst, count, ty, rank - size / 2, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Send(src, count, ty, rank - size / 2, 0, MPI_COMM_WORLD);
    }
    double stop = MPI_Wtime();

    // maximum time across all ranks
    double tmp = stop - start;
    MPI_Allreduce(MPI_IN_PLACE, &tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    stats.insert(tmp);
  }
  nvtxRangePop();

  if (host) {
    delete[] src;
    delete[] dst;
  } else {
    CUDA_RUNTIME(cudaFree(src));
    CUDA_RUNTIME(cudaFree(dst));
  }

  if (ty != MPI_BYTE) {
    MPI_Type_commit(&ty);
  }

  // type is send back and forth
  return BenchResult{.bytes = typeSize * count, .pingPongTime = stats.min()};
}

struct Factory1D {
  TypeFactory1D fn;
  const char *name;
};

struct Factory2D {
  TypeFactory2D fn;
  const char *name;
};

struct Factory3D {
  TypeFactory3D fn;
  const char *name;
};

int main(int argc, char **argv) {

  environment::noTempi = false;
  MPI_Init(&argc, &argv);

  // run on only ranks 0 and 1
  int size, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size % 2 != 0) {
    LOG_FATAL("needs even number of ranks");
  }

  int nIters = 10;
  std::string s;

  BenchResult result;

  std::vector<bool> hosts{true, false};
  std::vector<int> counts;

  /* 2D types
   */

  if (0 == rank) {
    std::cout << "desc,ranks,numblocks,blocklength,stride,B/rank,B,elapsed "
                 "(s),bandwidth/rank "
                 "(MiB/s), bandwidth agg (MiB/s)\n";
  }

  int numBlocks = 1000;
  int count = 1;
  std::vector<Factory2D> factories2d{
      Factory2D{make_2d_byte_vector, "2d_byte_vector"},
      Factory2D{make_2d_byte_hvector, "2d_byte_hvector"},
      Factory2D{make_2d_byte_subarray, "2d_byte_subarray"}};

  std::vector<int> blockLengths{1, 2, 4, 8, 128};

  for (Factory2D factory : factories2d) {

    for (int blockLength : blockLengths) {

      std::vector<int> strides;
      for (int i = blockLength; i < 512; i *= 2) {
        strides.push_back(i);
      }

      for (int stride : strides) {

        s = factory.name;
        s += "|" + std::to_string(numBlocks);
        s += "|" + std::to_string(blockLength);
        s += "|" + std::to_string(stride);

        if (0 == rank) {
          std::cout << s;
          std::cout << "," << size;
          std::cout << "," << numBlocks;
          std::cout << "," << blockLength;
          std::cout << "," << stride;
          std::cout << std::flush;
        }

        MPI_Datatype ty = factory.fn(numBlocks, blockLength, stride);

        result = bench(ty, 1, false /*host*/, nIters, s.c_str());

        if (0 == rank) {
          std::cout << "," << result.bytes;        // size of a send
          std::cout << "," << result.bytes * size; // total data sent
          // bw per rank. each pingpong has two sends
          std::cout << "," << 2 * result.bytes / result.pingPongTime;
          // agg bw (total data sent / time)
          std::cout << "," << result.bytes * size / result.pingPongTime;
          std::cout << "\n";
          std::cout << std::flush;
        }
      }
    }
  }

  /* 1D types
   */

  if (0 == rank) {
    std::cout << "desc,host,ranks,B/rank,B,elapsed (s),bandwidth/rank "
                 "(MiB/s), bandwidth agg (MiB/s)\n";
  }

  counts = {1,     2,       4,       8,       16,      32,      64,    128,
            256,   512,     1024,    1 << 11, 4096,    1 << 13, 16384, 1 << 15,
            65536, 1 << 17, 1 << 19, 1 << 20, 1 << 21, 1 << 24};

  for (bool host : hosts) {
    for (int count : counts) {

      s = std::to_string(host) + "|" + std::to_string(size) + "|" +
          std::to_string(count);

      if (0 == rank) {
        std::cout << s;
        std::cout << "," << host;
        std::cout << "," << size;
        std::cout << "," << count;
        std::cout << "," << count * size;
        std::cout << std::flush;
      }

      nvtxRangePush(s.c_str());
      result = bench(MPI_BYTE, count, host, nIters, s.c_str());
      nvtxRangePop();
      if (0 == rank) {
        std::cout << "," << result.pingPongTime;

        // half the ranks send, then the other half, so each rank sends once
        std::cout << "," << double(count) / 1024 / 1024 / result.pingPongTime;
        std::cout << ","
                  << double(count) * size / 1024 / 1024 / result.pingPongTime;
        std::cout << "\n";
        std::cout << std::flush;
      }
    }
  }

  MPI_Finalize();
  return 0;
}
