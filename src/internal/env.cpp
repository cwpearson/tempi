#include "env.hpp"

#include <cstdlib>

namespace environment {
/*extern*/ bool noPack;
/*extern*/ bool noTypeCommit;
}; // namespace environment

void read_environment() {
  using namespace environment;
  noPack = (nullptr == std::getenv("TEMPI_NO_PACK"));
  noTypeCommit = (nullptr == std::getenv("TEMPI_NO_TYPE_COMMIT"));
}