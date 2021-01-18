//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "numeric.hpp"

#include <mpi.h>

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

struct IidTime {
  double time;
  bool iid; // was the benchmark iid
};

/* The system benchmark code and interpolation code must match how they treat
 * this data
 */
struct SystemPerformance {
  double cudaKernelLaunch;

  // vec[i] is time for 2^i bytes
  std::vector<IidTime> intraNodeCpuCpuPingpong;
  std::vector<IidTime> intraNodeGpuGpuPingpong;
  std::vector<IidTime> interNodeCpuCpuPingpong;
  std::vector<IidTime> interNodeGpuGpuPingpong;
  std::vector<IidTime> d2h;
  std::vector<IidTime> h2d;

  /*vec[i][j] is 2^(2i+6) bytes with stride 2^j*/
  std::vector<std::vector<IidTime>> packDevice;
  std::vector<std::vector<IidTime>> unpackDevice;
  std::vector<std::vector<IidTime>> packHost;
  std::vector<std::vector<IidTime>> unpackHost;

  double guess_global_pack(int64_t bytes, int64_t stride);
};

extern SystemPerformance systemPerformance;

/* interpolate using a vector of bandwidth sorted by bytes
 */
double interp_time(const std::vector<IidTime> a, int64_t bytes);

/* interpolate using a vector of bandwidth sorted by bytes
 */
double interp_2d(const std::vector<std::vector<IidTime>> a, int64_t bytes,
                 int64_t stride);

// fill empty entries in sp
void measure_system_performance(SystemPerformance &sp, MPI_Comm comm);
bool export_system_performance(const SystemPerformance &sp);
bool import_system_performance(SystemPerformance &sp);

/*try to load system performance from file*/
void measure_system_init();