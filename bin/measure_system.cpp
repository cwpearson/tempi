//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "../include/measure_system.hpp"
#include "../include/logging.hpp"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    if (0 == rank) {
      std::cout << "please run with 2 ranks\n";
    }
    MPI_Finalize();
    return -1;
  }
  tempi::system::Performance sp{};
  // all ranks import existing measurements
  tempi::system::import_performance(sp);

  // complete possible missing entries
  tempi::system::measure_performance(sp, MPI_COMM_WORLD);

  // re-export
  if (0 == rank) {
    tempi::system::export_performance(sp);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
