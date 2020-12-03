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
/*extern*/ DatatypeMethod datatype;
}; // namespace environment

void read_environment() {
  using namespace environment;
  placement = PlacementMethod::NONE; // default to library placement
  alltoallv = AlltoallvMethod::AUTO;
  datatype = DatatypeMethod::ONESHOT;

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

  if (nullptr != std::getenv("TEMPI_PLACEMENT_METIS")) {
#ifdef TEMPI_ENABLE_METIS
    placement = PlacementMethod::METIS;
#else
    std::cerr << "ERROR: TEMPI_PLACEMENT_METIS in environment but TEMPI_ENABLE_METIS not defined\n";
#endif
  }
  if (nullptr != std::getenv("TEMPI_PLACEMENT_KAHIP")) {
#ifdef TEMPI_ENABLE_KAHIP
    placement = PlacementMethod::KAHIP;
#else
    std::cerr << "ERROR: TEMPI_PLACEMENT_KAHIP in environment but TEMPI_ENABLE_KAHIP not defined\n";
#endif
  }
  if (nullptr != std::getenv("TEMPI_PLACEMENT_RANDOM")) {
    placement = PlacementMethod::RANDOM;
  }

    if (nullptr != std::getenv("TEMPI_DATATYPE_ONESHOT")) {
    datatype = DatatypeMethod::ONESHOT;
  }
    if (nullptr != std::getenv("TEMPI_DATATYPE_DEVICE")) {
    datatype = DatatypeMethod::DEVICE;
  }
}
