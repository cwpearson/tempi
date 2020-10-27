#pragma once

enum class PlacementMethod {
  NONE,   // library placement
  RANDOM, // random placement of ranks
  METIS   // use metis to place ranks
};

namespace environment {
extern bool noTempi; // disable all TEMPI globally
extern bool noAlltoallv;
extern bool noPack;
extern bool noTypeCommit;
extern PlacementMethod placement;
}; // namespace environment

void read_environment();

#define TEMPI_DISABLE_GUARD                                                    \
  {                                                                            \
    if (environment::noTempi)                                                  \
      return fn(ARGS);                                                         \
  }
