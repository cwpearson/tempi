/*! \file
    measure various MPI methods for achiving the same communication pattern
*/

#include "../include/cuda_runtime.hpp"
#include "../include/env.hpp"
#include "../include/logging.hpp"

#include "benchmark.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>

int main(int argc, char **argv) {

  environment::noTempi = false; // enable at init
  MPI_Init(&argc, &argv);

  int size, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (0 == rank) {
    char version[MPI_MAX_LIBRARY_VERSION_STRING] = {};
    int len;
    MPI_Get_library_version(version, &len);
    std::cout << version << std::endl;
  }

  std::vector<BM::Pattern *> benchmarks{
      new BM::Pattern_alltoallv(), new BM::Pattern_isend_irecv(),
      new BM::Pattern_sparse_isend_irecv(),
      new BM::Pattern_reorder_neighbor_alltoallv()};

  MPI_Barrier(MPI_COMM_WORLD);

  int nIters = 30;

  std::vector<bool> tempis{true, false};
  std::vector<int64_t> scales{1,         10,         100,        1 * 1000,
                              10 * 1000, 100 * 1000, 1000 * 1000};
  std::vector<float> densities{1.0, 0.5, 0.1, 0.05};

  // add some more densities to target particular nnz per row
  for (int targetNnzPerRow : {1, 2, 4, 8, 16}) {
    float density = double(targetNnzPerRow) / size;
    if (density <= 1) {
      densities.push_back(density);
    }
  }

  if (0 == rank) {
    std::cout << "description,name,tempi,scale,density,B,min iter (s),agg iter "
                 "(MiB/s)\n";
  }

  for (bool tempi : tempis) {
    environment::noTempi = !tempi;

    for (BM::Pattern *benchmark : benchmarks) {

      for (int64_t scale : scales) {

        for (float density : densities) {

          SquareMat mat = SquareMat::make_random_sparse(
              size, size * density + 0.5, 1, 10, scale);
          int64_t numBytes = mat.reduce_sum();
          MPI_Bcast(mat.data_.data(), mat.data_.size(), MPI_INT, 0,
                    MPI_COMM_WORLD);

          std::string s;
          s = std::string(benchmark->name()) + "|" + std::to_string(tempi) +
              "|" + std::to_string(scale) + "|" + std::to_string(density);

          if (0 == rank) {
            std::cout << s;
            std::cout << "," << benchmark->name();
            std::cout << "," << tempi;
            std::cout << "," << scale;
            std::cout << "," << density;
            std::cout << std::flush;
          }

          nvtxRangePush(s.c_str());
          BM::Result result = (*benchmark)(mat, nIters);
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
  }

  for (auto &e : benchmarks) {
    delete e;
  }

  environment::noTempi = false; // enable since it was enabled at init
  MPI_Finalize();
  return 0;
}
