#include "test.hpp"

#include "../include/numeric.hpp"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

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

  return 0;
}