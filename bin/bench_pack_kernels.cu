//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "statistics.hpp"

#include "../include/allocators.hpp"
#include "../include/cuda_runtime.hpp"
#include "../include/pack_kernels.cuh"

#include <nvToolsExt.h>

#include <chrono>
#include <cstring> //memset
#include <iostream>
#include <sstream>

typedef std::chrono::system_clock Clock;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::time_point<Clock, Duration> Time;

struct BenchResult {
  int64_t size;
  double packTime;
  double unpackTime;
};

struct BenchArgs {
  int64_t numBlocks;
  int64_t blockLength;
  int64_t stride;
  int64_t count; // number of objects
};

BenchResult bench(const BenchArgs &args, // message datatype
                  const int nIters,      // iterations to measure
                  const bool stage,      // pack into host / unpack from host
                  const char *name = "<unnamed>") {

  int64_t objExt = (args.numBlocks - 1) * args.stride + args.blockLength;
  const int64_t packedSize = args.numBlocks * args.blockLength;

  char *src{}, *dst{};
  CUDA_RUNTIME(cudaSetDevice(0));
  CUDA_RUNTIME(cudaMalloc(&src, objExt * args.count));
  if (stage) {
    dst = hostAllocator.allocate(packedSize * args.count);
  } else {
    CUDA_RUNTIME(cudaMalloc(&dst, packedSize * args.count));
  }

  if (stage) {
    CUDA_RUNTIME(cudaMemset(src, 0xFE, objExt * args.count));
    std::memset(dst, 0x00, packedSize * args.count);
  }

  cudaStream_t stream;
  cudaEvent_t start, stop;
  CUDA_RUNTIME(cudaStreamCreate(&stream));
  CUDA_RUNTIME(cudaEventCreate(&start));
  CUDA_RUNTIME(cudaEventCreate(&stop));

  Pack2DConfig config(args.blockLength, args.numBlocks);

  dim3 gd = config.dim_grid(args.count);
  dim3 bd = config.dim_block();

  // dimBlock = 32;
  // dimGrid = 1;

  std::cerr << " [" << gd.x << " " << gd.y << " " << gd.z << "]x[" << bd.x
            << " " << bd.y << " " << bd.z << "] ";

#if 0
  std::cerr << "[" << dimGrid.x << " " << dimGrid.y << " " << dimGrid.z <<
  std::endl;
#endif

  Statistics packStats;
  nvtxRangePush(name);
  for (int n = 0; n < nIters; ++n) {

    CUDA_RUNTIME(cudaEventRecord(start, stream));
    config.packfn<<<gd, bd, 0, stream>>>(dst, src, args.count, args.blockLength,
                                         args.numBlocks, args.stride, objExt);
    CUDA_RUNTIME(cudaEventRecord(stop, stream));
    CUDA_RUNTIME(cudaEventSynchronize(stop));
    CUDA_RUNTIME(cudaGetLastError());
    float millis;
    CUDA_RUNTIME(cudaEventElapsedTime(&millis, start, stop));
    packStats.insert(millis / 1024.0);
  }
  nvtxRangePop();

  if (stage) {
    for (size_t i = 0; i < packedSize * args.count; ++i) {
      if (dst[i] != char(0xFE)) {
        exit(-1);
      }
    }
  }

  CUDA_RUNTIME(cudaFree(src));
  if (stage) {
    hostAllocator.deallocate(dst, packedSize * args.count);
  } else {
    CUDA_RUNTIME(cudaFree(dst));
  }

  CUDA_RUNTIME(cudaEventDestroy(start));
  CUDA_RUNTIME(cudaEventDestroy(stop));
  CUDA_RUNTIME(cudaStreamDestroy(stream));

  return BenchResult{.size = packedSize * args.count,
                     .packTime = packStats.trimean(),
                     .unpackTime = 0};
}

int main(int argc, char **argv) {

  int nIters = 30;

  std::vector<bool> stages{
      false, true}; // whether to one-shot pack device-host / unpack host-device

  BenchResult result;

  /* 2D packing
   */

  std::vector<int> targets{1024, 1024 * 1024};
  // targets = {1024 * 1024};

  std::vector<int> counts{1, 2};
  // counts = {1};

  std::cout << "s,one-shot,count,obj size(B),stride,blocklengths,s,MiB/s";
  std::cout << std::endl << std::flush;

  std::vector<int> blockLengths{1,  2,  4,  6,  8,  12,  16, 20,
                                24, 28, 32, 64, 96, 128, 256};
  // blockLengths = {1};
  std::vector<int> strides{16, 256};
  // strides = {16};

  for (bool stage : stages) {
    for (int target : targets) {
      for (int count : counts) {
        for (int stride : strides) {
          for (int blockLength : blockLengths) {

            int numBlocks = target / blockLength;

            if (numBlocks > 0 && stride >= blockLength) {

              std::string s;
              s += std::to_string(stage);
              s += "|" + std::to_string(count);
              s += "|" + std::to_string(target);
              s += "|" + std::to_string(stride);
              s += "|" + std::to_string(blockLength);

              std::cout << s;
              std::cout << "," << stage;
              std::cout << "," << count;
              std::cout << "," << target;
              std::cout << "," << stride;
              std::cout << "," << blockLength;
              std::cout << std::flush;

              BenchArgs args{.numBlocks = numBlocks,
                             .blockLength = blockLength,
                             .stride = stride,
                             .count = count};

              result = bench(args, nIters, stage, s.c_str());
              std::cout << "," << result.packTime;
              std::cout << ","
                        << double(result.size) / 1024.0 / 1024.0 /
                               result.packTime;
              std::cout << std::flush;
              std::cout << std::endl << std::flush;
            }
          }
        }
      }
    }
  }

  return 0;
}
