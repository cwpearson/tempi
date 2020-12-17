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
  SystemPerformance sp{};
  // all ranks import existing measurements
  import_system_performance(sp);

  // complete possible missing entries
  measure_system_performance(sp, MPI_COMM_WORLD);

  // re-export
  if (0 == rank) {
    export_system_performance(sp);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
