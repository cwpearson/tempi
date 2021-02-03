//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "martinmoene/optional.hpp"
#include "numeric.hpp"

#include <mpi.h>

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

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

  // device-to-host time
  nonstd::optional<double> contig_d2h(int64_t bytes) const;

  // non-contiguous pack and unpack into host memory
  nonstd::optional<double> pack_host_noncontig(int64_t bytes,
                                               int64_t blockLength) const;
  nonstd::optional<double> unpack_host_noncontig(int64_t bytes,
                                                 int64_t blockLength) const;

  // non-contiguous pack and unpack into device memory
  nonstd::optional<double> pack_device_noncontig(int64_t bytes,
                                                 int64_t blockLength) const;
  nonstd::optional<double> unpack_device_noncontig(int64_t bytes,
                                                   int64_t blockLength) const;

  // contiguous cpu-cpu/gpu-gpu send
  nonstd::optional<double> send_h2h_contig(bool colocated, int64_t bytes) const;
  nonstd::optional<double> send_d2d_contig(bool colocated, int64_t bytes) const;

  // Data for send non-contiguos n-dimensional data
  struct SendNonContigNd {
    struct Args {
      bool colocated;
      int64_t bytes;
      int64_t blockLength;
      struct Hasher { // unordered_map key
        size_t operator()(const Args &a) const noexcept {
          return std::hash<bool>()(a.colocated) ^
                 std::hash<int64_t>()(a.bytes) ^
                 std::hash<int64_t>()(a.blockLength);
        }
      };
      bool operator==(const Args &rhs) const { // unordered_map key
        return colocated == rhs.colocated && bytes == rhs.bytes &&
               blockLength == rhs.blockLength;
      }
    };

    enum class Method { UNKNOWN, DEVICE, ONESHOT };

    typedef std::unordered_map<Args, Method, Args::Hasher> MethodCache;
  };

  // send non-contiguos n-dimensional data
  nonstd::optional<double>
  model_oneshot(const SendNonContigNd::Args &args) const;
  nonstd::optional<double>
  model_device(const SendNonContigNd::Args &args) const;

#if 0
  // non-contiguous ndimensional one-shot send
  nonstd::optional<double> send_noncontig_oneshot_nd(bool colocated,
                                                     int64_t bytes,
                                                     int64_t blockLength) const;
  // non-contiguous ndimensional device send
  nonstd::optional<double> send_noncontig_device_nd(bool colocated,
                                                    int64_t bytes,
                                                    int64_t blockLength) const;
#endif
};

extern SystemPerformance systemPerformance;

/* interpolate using a vector of bandwidth sorted by bytes.
non-optional version returns infinity on unknown
 */
double interp_time(const std::vector<IidTime> a, int64_t bytes);
nonstd::optional<double> interp_time_opt(const std::vector<IidTime> a,
                                         int64_t bytes);

/* interpolate using a vector of bandwidth sorted by bytes
 */
double interp_2d(const std::vector<std::vector<IidTime>> a, int64_t bytes,
                 int64_t stride);
nonstd::optional<double>
interp_2d_opt(const std::vector<std::vector<IidTime>> a, int64_t bytes,
              int64_t stride);

// fill empty entries in sp
void measure_system_performance(SystemPerformance &sp, MPI_Comm comm);
bool export_system_performance(const SystemPerformance &sp);
bool import_system_performance(SystemPerformance &sp);

/*try to load system performance from file*/
void measure_system_init();