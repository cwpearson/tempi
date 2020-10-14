#include "env.hpp"

#include <cstdlib>
#include <iostream>

namespace environment {
/*extern*/ bool noTempi;
/*extern*/ bool noPack;
/*extern*/ bool noTypeCommit;
}; // namespace environment

void read_environment() {
  using namespace environment;
  noTempi = (nullptr != std::getenv("TEMPI_DISABLE"));
  noPack = (nullptr != std::getenv("TEMPI_NO_PACK"));
  noTypeCommit = (nullptr != std::getenv("TEMPI_NO_TYPE_COMMIT"));
}