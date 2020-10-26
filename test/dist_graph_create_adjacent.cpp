#include <mpi.h>

#include "../support/type.hpp"

#include "test.hpp"

#include <vector>

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // pairs of
  std::vector<int> sources(1);
  std::vector<int> destinations(1);
  std::vector<int> sourceweights(sources.size(), 1);
  std::vector<int> destweights(destinations.size(), 1);

  sources[0] = rank + size / 2;
  destinations[0] = rank + size / 2;
  if (sources[0] >= size) {
    sources[0] %= size;
  }
  if (destinations[0] >= size) {
    destinations[0] %= size;
  }

  MPI_Comm graph{};

  MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, sources.size(), sources.data(),
                                 sourceweights.data(), destinations.size(),
                                 destinations.data(), destweights.data(),
                                 MPI_INFO_NULL, 1, &graph);

  MPI_Comm_free(&graph);

  MPI_Finalize();

  return 0;
}