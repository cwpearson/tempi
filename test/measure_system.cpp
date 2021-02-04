//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

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

  // clang-format off
  std::vector<std::vector<IidTime>> a{
             /*1*/        /*2*/        /*4*/
    /*64*/  {{14, false}, {18, false}, {22, false}    },
    /*256*/ {{16, false}, {20, false}, {24, false}    }
    };
  // clang-format on

  {
    double r = interp_2d(a, 64, 1);
    REQUIRE(r == 14);
  }
  {
    double r = interp_2d(a, 160, 1);

    REQUIRE(r == 15);
  }
  {
    double r = interp_2d(a, 256, 1);
    REQUIRE(r == 16);
  }
  {
    double r = interp_2d(a, 64, 2);
    REQUIRE(r == 18);
  }
  {
    double r = interp_2d(a, 256, 2);
    REQUIRE(r == 20);
  }
  {
    double r = interp_2d(a, 64, 2);
    REQUIRE(r == 18);
  }
  {
    double r = interp_2d(a, 64, 3);
    REQUIRE(r == 20);
  }
  {
    double r = interp_2d(a, 64, 4);
    REQUIRE(r == 22);
  }
  {
    double r = interp_2d(a, 160, 3);
    REQUIRE(r == 21);
  }
  {
    double r = interp_2d(a, 512, 1);
    REQUIRE(r == 32);
  }
  {
    double r = interp_2d(a, 512, 3);
    REQUIRE(r == 44);
  }

  MPI_Finalize();

  return 0;
}