#pragma once

#include <mpi.h>

#include <string>
#include <vector>

struct Bandwidth {
  int64_t bytes;
  double time;
  bool iid;
};

struct SystemPerformance {
  double cudaKernelLaunch;
  std::vector<Bandwidth> intraNodeCpuCpuPingpong;
  std::vector<Bandwidth> intraNodeGpuGpuPingpong;
  std::vector<Bandwidth> interNodeCpuCpuPingpong;
  std::vector<Bandwidth> interNodeGpuGpuPingpong;
  std::vector<Bandwidth> d2h;
  std::vector<Bandwidth> h2d;
};

// fill empty entries in sp
void measure_system_performance(SystemPerformance &sp, MPI_Comm comm);
bool export_system_performance(const SystemPerformance &sp);
bool import_system_performance(SystemPerformance &sp);
