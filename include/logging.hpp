#pragma once

#include <mpi.h>

#include <cstdlib>
#include <iostream>

#ifndef TEMPI_OUTPUT_LEVEL
#define TEMPI_OUTPUT_LEVEL 3
#endif

inline int world_rank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

#if TEMPI_OUTPUT_LEVEL >= 5
#define LOG_SPEW(x)                                                                                                    \
  std::cerr << "SPEW[" << __FILE__ << ":" << __LINE__ << "]{" << world_rank() << "} " << x << "\n";
#else
#define LOG_SPEW(x)
#endif

#if TEMPI_OUTPUT_LEVEL >= 4
#define LOG_DEBUG(x)                                                                                                   \
  std::cerr << "DEBUG[" << __FILE__ << ":" << __LINE__ << "]{" << world_rank() << "} " << x << "\n";
#else
#define LOG_DEBUG(x)
#endif

#if TEMPI_OUTPUT_LEVEL >= 3
#define LOG_INFO(x)                                                                                                    \
  std::cerr << "INFO[" << __FILE__ << ":" << __LINE__ << "]{" << world_rank() << "} " << x << "\n";
#else
#define LOG_INFO(x)
#endif

#if TEMPI_OUTPUT_LEVEL >= 2
#define LOG_WARN(x)                                                                                                    \
  std::cerr << "WARN[" << __FILE__ << ":" << __LINE__ << "]{" << world_rank() << "} " << x << "\n";
#else
#define LOG_WARN(x)
#endif

#if TEMPI_OUTPUT_LEVEL >= 1
#define LOG_ERROR(x)                                                                                                   \
  std::cerr << "ERROR[" << __FILE__ << ":" << __LINE__ << "]{" << world_rank() << "} " << x << "\n";
#else
#define LOG_ERROR(x)
#endif

#if TEMPI_OUTPUT_LEVEL >= 0
#define LOG_FATAL(x)                                                                                                   \
  std::cerr << "FATAL[" << __FILE__ << ":" << __LINE__ << "]{" << world_rank() << "} " << x << "\n";              \
  exit(1);
#else
#define LOG_FATAL(x) exit(1);
#endif