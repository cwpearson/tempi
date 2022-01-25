//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "test.hpp"

#include "../include/numeric.hpp"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (1 != size) {
    std::cerr << "ERROR: requires exactly 1 rank\n";
    exit(1);
  }


  REQUIRE(log2_ceil(1) == 0);
  REQUIRE(log2_floor(1) == 0);

  REQUIRE(log2_ceil(2) == 1);
  REQUIRE(log2_floor(2) == 1);

  REQUIRE(log2_ceil(3) == 2);
  REQUIRE(log2_floor(3) == 1);

  REQUIRE(log2_ceil(4) == 2);
  REQUIRE(log2_floor(4) == 2);

  REQUIRE(log2_ceil(5) == 3);
  REQUIRE(log2_floor(5) == 2);

  MPI_Finalize();
  return 0;
}