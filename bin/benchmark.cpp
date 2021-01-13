//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "benchmark.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <algorithm>
#include <iostream>

/*  run a variety of different random communication pattern benchmarks on the
 * given method
 */
void random(BM::Method *method, const std::vector<int64_t> &scales_,
            const std::vector<float> densities_, const int nIters) {

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int64_t> scales = scales_;
  std::vector<float> densities = densities_;

  std::sort(scales.begin(), scales.end());
  std::sort(densities.begin(), densities.end());

  for (int64_t scale : scales) {

    for (float density : densities) {

      SquareMat mat = SquareMat::make_random_sparse(size, size * density + 0.5,
                                                    1, 10, scale);
      int64_t numBytes = mat.reduce_sum();
      MPI_Bcast(mat.data_.data(), mat.data_.size(), MPI_INT, 0, MPI_COMM_WORLD);

      std::string s;
      s = std::string(method->name()) + "|" + std::to_string(scale) + "|" +
          std::to_string(density);

      if (0 == rank) {
        std::cout << s;
        std::cout << "," << method->name();
        std::cout << "," << scale;
        std::cout << "," << density;
        std::cout << std::flush;
      }

      nvtxRangePush(s.c_str());
      BM::Result result = (*method)(mat, nIters);
      nvtxRangePop();
      if (0 == rank) {
        std::cout << "," << numBytes << "," << result.iters.min() << ","
                  << double(numBytes) / 1024 / 1024 / result.iters.min();
        std::cout << std::flush;
      }

      if (0 == rank) {
        std::cout << "\n";
        std::cout << std::flush;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
}

const char *random_csv_header() {
  return "description,name,scale,density,B,min iter (s),agg iter (MiB/s)\n";
}
