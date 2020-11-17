#include "../include/cuda_runtime.hpp"
#include "../include/env.hpp"
#include "../include/logging.hpp"
#include "../support/type.hpp"
#include "statistics.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <chrono>
#include <sstream>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::time_point<Clock, Duration> Time;

struct BenchResult {
  double typeCreateTime;
  double typeCommitTime;
};

template <typename Factory>
BenchResult bench(Factory factory, const Dim3 &copyExt, const Dim3 &allocExt,
                  const int nIters) {

  Statistics commitStats, createStats;
  for (int n = 0; n < nIters; ++n) {
    MPI_Datatype ty;
    {
      auto start = Clock::now();
      ty = factory(copyExt, allocExt);
      auto stop = Clock::now();
      Duration elapsed = stop - start;
      createStats.insert(elapsed.count());
    }
    {
      auto start = Clock::now();
      MPI_Type_commit(&ty);
      auto stop = Clock::now();
      Duration elapsed = stop - start;
      commitStats.insert(elapsed.count());
    }
    MPI_Type_free(&ty);
  }

  return BenchResult{.typeCreateTime = createStats.trimean(),
                     .typeCommitTime = commitStats.trimean()};
}

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  // run on only ranks 0 and 1
  int size, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 1) {
    if (0 == rank) {
      LOG_FATAL("needs only one rank");
    } else {
      exit(-1);
    }
  }

  { // prevent init bypass
    int nIters = 30000;

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

    if (0 == rank) {
      std::cout << "s,x,y,z";
      std::cout << ",subarray [create] (s),subarray [commit] (s)";
      std::cout << ",byte_v_hv [create] (s),byte_v_hv [commit] (s)";
      std::cout << ",byte_v1_hv_hv [create] (s),byte_v1_hv_hv [commit] (s)";
      std::cout << ",byte_vn_hv_hv [create] (s),byte_v1_hv_hv [commit] (s)";
      std::cout << ",subarray_v [create] (s),byte_v1_hv_hv [commit] (s)";
      std::cout << std::endl;
    }

    for (Dim3 ext : dims) {

      std::string s;
      s = std::to_string(ext.x) + "|" + std::to_string(ext.y) + "|" +
          std::to_string(ext.z);

      if (0 == rank) {
        std::cout << s;
        std::cout << "," << ext.x << "," << ext.y << "," << ext.z;
        std::cout << std::flush;
      }

#if 1
      result = bench(make_subarray, ext, allocExt, nIters);
      if (0 == rank) {
        std::cout << "," << result.typeCreateTime;
        std::cout << "," << result.typeCommitTime;
        std::cout << std::flush;
      }
#endif

#if 1
      result = bench(make_byte_v_hv, ext, allocExt, nIters);
      if (0 == rank) {
        std::cout << "," << result.typeCreateTime;
        std::cout << "," << result.typeCommitTime;
        std::cout << std::flush;
      }
#endif

#if 1
      result = bench(make_byte_v1_hv_hv, ext, allocExt, nIters);
      if (0 == rank) {
        std::cout << "," << result.typeCreateTime;
        std::cout << "," << result.typeCommitTime;
        std::cout << std::flush;
      }
#endif

#if 1
      result = bench(make_byte_vn_hv_hv, ext, allocExt, nIters);
      if (0 == rank) {
        std::cout << "," << result.typeCreateTime;
        std::cout << "," << result.typeCommitTime;
        std::cout << std::flush;
      }
#endif

#if 1
      result = bench(make_subarray_v, ext, allocExt, nIters);
      if (0 == rank) {
        std::cout << "," << result.typeCreateTime;
        std::cout << "," << result.typeCommitTime;
        std::cout << std::flush;
      }
#endif

      if (0 == rank) {
        std::cout << "\n";
        std::cout << std::flush;
      }
      nvtxRangePop();
    }
  }
finalize:
  MPI_Finalize();
  return 0;
}
