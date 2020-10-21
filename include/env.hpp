#pragma once

namespace environment {
extern bool noTempi; // disable all TEMPI globally
extern bool noAlltoallv;
extern bool noPack;
extern bool noTypeCommit;
}; // namespace environment

void read_environment();

#define TEMPI_DISABLE_GUARD                                                    \
  {                                                                            \
    if (environment::noTempi)                                                  \
      return fn(ARGS);                                                         \
  }
