#include "env.hpp"

#include <cstdlib>

namespace environment {
    /*extern*/ bool noPack;
};

void read_environment() {
    using namespace environment;
    
    noPack = (nullptr == std::getenv("SCAMPI_NO_PACK"));
}