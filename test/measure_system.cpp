#include "test.hpp"

#include "../include/measure_system.hpp"

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  measure_system(MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}