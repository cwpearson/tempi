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
  AUTO,
  ONESHOT, // pack into mapped host buffer and send
  DEVICE,  // pack into device buffer and send
  STAGED   // pack into device buffer, copy to host, and send
};

enum class ContiguousMethod {
  AUTO,
  NONE,  // use library
  STAGED // copy to host, and send
};

namespace environment {
extern bool noTempi; // disable all TEMPI globally
extern bool noPack;
extern bool noTypeCommit;
extern AlltoallvMethod alltoallv;
extern DatatypeMethod datatype;
extern PlacementMethod placement;
extern ContiguousMethod contiguous;
extern std::string cacheDir;
}; // namespace environment

void read_environment();

#define TEMPI_DISABLE_GUARD                                                    \
  {                                                                            \
    if (environment::noTempi)                                                  \
      return fn(ARGS);                                                         \
  }
