#include "test.hpp"

#include "../include/measure_system.hpp"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  std::vector<IidTime> v{{2, false}, {4, false}, {8, false}, {16, false}};

  {
    double r = interp_time(v, 1);
    REQUIRE(r == 2);
  }

  {
    double r = interp_time(v, 2);
    REQUIRE(r == 4);
  }

  {
    double r = interp_time(v, 3);
    REQUIRE(r == 6);
  }

  {
    double r = interp_time(v, 5);
    REQUIRE(r == 10);
  }

  {
    double r = interp_time(v, 7);
    REQUIRE(r == 14);
  }

  MPI_Finalize();

  return 0;
}