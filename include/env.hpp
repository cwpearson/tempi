#pragma once

#include <string>

enum class PlacementMethod {
  NONE,   // library placement
  RANDOM, // random placement of ranks
  METIS,  // use METIS to place ranks
  KAHIP   // use KaHIP to place ranks
};

enum class AlltoallvMethod {
  NONE, // use library MPI_Alltoallv
  AUTO,
  REMOTE_FIRST,
  STAGED,
  ISIR_STAGED,
  ISIR_REMOTE_STAGED
};

enum class DatatypeMethod {
  ONESHOT, // pack into mapped host buffer
  DEVICE   // pack into device buffer
};

namespace environment {
extern bool noTempi; // disable all TEMPI globally
extern bool noPack;
extern bool noTypeCommit;
extern PlacementMethod placement;
extern AlltoallvMethod alltoallv;
extern DatatypeMethod datatype;
extern std::string cacheDir;
}; // namespace environment

void read_environment();

#define TEMPI_DISABLE_GUARD                                                    \
  {                                                                            \
    if (environment::noTempi)                                                  \
      return fn(ARGS);                                                         \
  }
