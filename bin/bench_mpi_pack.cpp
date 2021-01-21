//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "../include/cuda_runtime.hpp"
#include "../include/env.hpp"
#include "../support/type.hpp"
#include "statistics.hpp"
#include "../include/allocators.hpp"

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
                  int count,        // number of datatypes
                  const int nIters, // iterations to measure
                  const bool stage, // pack into host / unpack from host
                  const char *name = "<unnamed>") {

  MPI_Type_commit(&ty);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Aint lb, typeExtent;
  MPI_Type_get_extent(ty, &lb, &typeExtent);
  int typeSize;
  MPI_Type_size(ty, &typeSize);
  int packedSize;
  MPI_Pack_size(count, ty, MPI_COMM_WORLD, &packedSize);

  char *src = {}, *dst = {};
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc(&src, typeExtent * count));
  if (stage) {
    dst = hostAllocator.allocate(packedSize);
  } else {
    CUDA_RUNTIME(cudaMalloc(&dst, packedSize));
  }

  Statistics packStats, unpackStats;
  nvtxRangePush(name);
  for (int n = 0; n < nIters; ++n) {
    int position = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    MPI_Pack(src, count, ty, dst, packedSize, &position, MPI_COMM_WORLD);
    double stop = MPI_Wtime();
    packStats.insert(stop - start);

    position = 0;
    start = MPI_Wtime();
    MPI_Unpack(dst, packedSize, &position, src, count, ty, MPI_COMM_WORLD);
    stop = MPI_Wtime();
    unpackStats.insert(stop - start);
  }
  nvtxRangePop();

  CUDA_RUNTIME(cudaFree(src));
  if (stage) {
    hostAllocator.deallocate(dst, packedSize);
  } else {
    CUDA_RUNTIME(cudaFree(dst));
  }

  MPI_Type_free(&ty);

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
    nIters = 500;
  }

  std::vector<bool> stage{
      true}; // whether to one-shot pack device-host / unpack host-device

  BenchResult result;

  /* 2D packing
   */

  std::vector<int> targets{1024, 1024 * 1024};

  std::vector<int> counts{1, 2};

  std::vector<Factory2D> factories2d{
      Factory2D{make_2d_byte_vector, "2d_byte_vector"},
      Factory2D{make_2d_byte_hvector, "2d_byte_hvector"},
      Factory2D{make_2d_byte_subarray, "2d_byte_subarray"}};

  std::cout << "s,count,numblocks,stride,blocklengths";
  for (Factory2D factory : factories2d) {
    std::cout << "," << factory.name << " (MiB/s)";
  }
  std::cout << std::endl << std::flush;

  std::vector<int> blockLengths{1, 2, 4, 8, 32, 64, 128, 256, 512};
  std::vector<int> strides{512};

  for (int target : targets) {
    for (int count : counts) {
      for (int stride : strides) {
        for (int blockLength : blockLengths) {

          int numBlocks = target / blockLength;

          if (numBlocks > 0 && stride >= blockLength) {

            std::string s;
            s += std::to_string(count);
            s += "|" + std::to_string(target);
            s += "|" + std::to_string(stride);
            s += "|" + std::to_string(blockLength);

            std::cout << s;
            std::cout << "," << count;
            std::cout << "," << target;
            std::cout << "," << stride;
            std::cout << "," << blockLength;
            std::cout << std::flush;
            for (Factory2D factory : factories2d) {

              MPI_Datatype ty = factory.fn(numBlocks, blockLength, stride);

              result = bench(ty, count, nIters, false, s.c_str());

              std::cout << ","
                        << double(result.size) / 1024.0 / 1024.0 /
                               result.packTime;
              std::cout << std::flush;
            }
            std::cout << std::endl << std::flush;
          }
        }
      }
    }
  }

  /* 3D packing
   */
  Dim3 allocExt(1024, 1024, 1024);

  std::vector<Dim3> dims{
      /*fixed size 1M*/
      Dim3(1, 1024, 1024), Dim3(2, 1024, 512), Dim3(4, 1024, 256),
      Dim3(8, 1024, 128), Dim3(16, 1024, 64), Dim3(32, 1024, 32),
      Dim3(64, 1024, 16), Dim3(128, 1024, 8), Dim3(256, 1024, 4),
      Dim3(512, 1024, 2), Dim3(1024, 1024, 1),
      /*sweep 1k->512K*/
      Dim3(1, 1024, 1), Dim3(2, 1024, 1), Dim3(4, 1024, 1), Dim3(8, 1024, 1),
      Dim3(16, 1024, 1), Dim3(32, 1024, 1), Dim3(64, 1024, 1),
      Dim3(128, 1024, 1), Dim3(256, 1024, 1), Dim3(512, 1024, 1),
      /*stencil halos*/
      Dim3(12, 512, 512), Dim3(512, 3, 512), Dim3(512, 512, 3)};

  std::vector<Factory3D> factories3d{
      Factory3D{make_subarray, "subarray"},
      Factory3D{make_subarray_v, "subarray_v"},
      Factory3D{make_byte_v1_hv_hv, "byte_v1_hv_hv"},
      Factory3D{make_byte_vn_hv_hv, "byte_vn_hv_hv"},
      Factory3D{make_byte_v_hv, "byte_v_hv"}};

  std::cout << "s,x,y,z";
  for (Factory3D factory : factories3d) {
    std::cout << "," << factory.name << " (MiB/s)";
  }
  std::cout << std::endl << std::flush;

  counts = {1, 2};

  for (int count : counts) {
    for (Dim3 ext : dims) {

      std::string s;
      s += std::to_string(count);
      s += "|" + std::to_string(ext.x);
      s += "|" + std::to_string(ext.y);
      s += "|" + std::to_string(ext.z);

      std::cout << s;
      std::cout << "," << count;
      std::cout << "," << ext.x << "," << ext.y << "," << ext.z;
      std::cout << std::flush;

      for (Factory3D factory : factories3d) {

        MPI_Datatype ty = factory.fn(ext, allocExt);
        result = bench(ty, count, nIters, s.c_str());

        std::cout << "," << result.size / 1024.0 / 1024.0 / result.packTime;
        std::cout << std::flush;

        nvtxRangePop();
      }
      std::cout << std::endl << std::flush;
    }
  }

  MPI_Finalize();
  return 0;
}
