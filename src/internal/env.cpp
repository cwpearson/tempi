#include "env.hpp"

#include <cstdlib>
#include <iostream>

namespace environment {
/*extern*/ bool noTempi;
/*extern*/ bool noAlltoallv;
/*extern*/ bool noPack;
/*extern*/ bool noTypeCommit;
/*extern*/ PlacementMethod placement;
/*extern*/ AlltoallvMethod alltoallv;
}; // namespace environment

void read_environment() {
  using namespace environment;
  placement = PlacementMethod::NONE; // default to library placement
  alltoallv = AlltoallvMethod::AUTO;

  noTempi = (nullptr != std::getenv("TEMPI_DISABLE"));
  noPack = (nullptr != std::getenv("TEMPI_NO_PACK"));
  noTypeCommit = (nullptr != std::getenv("TEMPI_NO_TYPE_COMMIT"));

  if (nullptr != std::getenv("TEMPI_ALLTOALLV_REMOTE_FIRST")) {
    alltoallv = AlltoallvMethod::REMOTE_FIRST;
  }
  if (nullptr != std::getenv("TEMPI_ALLTOALLV_STAGED")) {
    alltoallv = AlltoallvMethod::STAGED;
  }
  if (nullptr != std::getenv("TEMPI_ALLTOALLV_ISIR_STAGED")) {
    alltoallv = AlltoallvMethod::ISIR_STAGED;
  }
  if (nullptr != std::getenv("TEMPI_ALLTOALLV_ISIR_REMOTE_STAGED")) {
    alltoallv = AlltoallvMethod::ISIR_REMOTE_STAGED;
  }
  if (nullptr != std::getenv("TEMPI_NO_ALLTOALLV")) {
    alltoallv = AlltoallvMethod::NONE;
  }

#ifdef TEMPI_ENABLE_METIS
  if (nullptr != std::getenv("TEMPI_PLACEMENT_METIS")) {
    placement = PlacementMethod::METIS;
  }
#endif
#ifdef TEMPI_ENABLE_KAHIP
  if (nullptr != std::getenv("TEMPI_PLACEMENT_KAHIP")) {
    placement = PlacementMethod::KAHIP;
  }
#endif
  if (nullptr != std::getenv("TEMPI_PLACEMENT_RANDOM")) {
    placement = PlacementMethod::RANDOM;
  }
}