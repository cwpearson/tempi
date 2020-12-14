#pragma once

#include "numeric.hpp"

#include <mpi.h>

#include <iostream>
#include <string>
#include <vector>

struct IidTime {
  double time;
  bool iid;
};

struct SystemPerformance {
  double cudaKernelLaunch;

  // vec[i] is time for 2^i bytes
  std::vector<IidTime> intraNodeCpuCpuPingpong;
  std::vector<IidTime> intraNodeGpuGpuPingpong;
  std::vector<IidTime> interNodeCpuCpuPingpong;
  std::vector<IidTime> interNodeGpuGpuPingpong;
  std::vector<IidTime> d2h;
  std::vector<IidTime> h2d;
};

extern SystemPerformance systemPerformance;

/* interpolate using a vector of bandwidth sorted by bytes
 */
inline double interp_time(const std::vector<IidTime> a, int64_t bytes) {
  uint8_t lb = log2_floor(bytes);
  uint8_t ub = log2_ceil(bytes);

  // too large, just scale up the largest time
  if (ub >= a.size()) {
    return a.back().time * bytes / (1ull << (a.size() - 1));
  } else if (lb == ub) {
    return a[lb].time;
  } else { // interpolate between points
    float num = bytes - (1ull << lb);
    float den = (1ull << ub) - (1ull << lb);
    float sf = num / den;
    return a[lb].time * (1 - sf) + a[ub].time * sf;
  }
}

// fill empty entries in sp
void measure_system_performance(SystemPerformance &sp, MPI_Comm comm);
bool export_system_performance(const SystemPerformance &sp);
bool import_system_performance(SystemPerformance &sp);

/*try to load system performance from file*/
void measure_system_init();