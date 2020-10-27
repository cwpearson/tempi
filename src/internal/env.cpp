#include "env.hpp"

#include <cstdlib>
#include <iostream>

namespace environment {
/*extern*/ bool noTempi;
/*extern*/ bool noAlltoallv;
/*extern*/ bool noPack;
/*extern*/ bool noTypeCommit;
/*extern*/ PlacementMethod placement;
}; // namespace environment

void read_environment() {
  using namespace environment;
  placement = PlacementMethod::NONE; // default to library placement

  noTempi = (nullptr != std::getenv("TEMPI_DISABLE"));
  noAlltoallv = (nullptr != std::getenv("TEMPI_NO_ALLTOALLV"));
  noPack = (nullptr != std::getenv("TEMPI_NO_PACK"));
  noTypeCommit = (nullptr != std::getenv("TEMPI_NO_TYPE_COMMIT"));

  if (nullptr != std::getenv("TEMPI_PLACEMENT_METIS")) {
    placement = PlacementMethod::METIS;
  };
  if (nullptr != std::getenv("TEMPI_PLACEMENT_RANDOM")) {
    placement = PlacementMethod::RANDOM;
  }
}