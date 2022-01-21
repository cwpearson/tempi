//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

/* some problem with using nlohmann::json in CUDA 11, so all
   the json-related stuff has to happen here
*/

#include "measure_system.hpp"

#include "env.hpp"
#include "logging.hpp"

#include "nlohmann/json.hpp"
using json = nlohmann::json;

#include <filesystem>
#include <fstream>



namespace tempi {
namespace system {

void to_json(json &j, const tempi::system::IidTime &p) {
  j["time"] = p.time;
  j["iid"] = p.iid;
}
void from_json(const json &j, tempi::system::IidTime &p) {
  j.at("time").get_to(p.time);
  j.at("iid").get_to(p.iid);
}

// === SystemPerformance ===
void to_json(json &j, const tempi::system::Performance &p) {
  j["intraNodeCpuCpuPingpong"] = p.intraNodeCpuCpuPingpong;
  j["intraNodeGpuGpuPingpong"] = p.intraNodeGpuGpuPingpong;
  j["interNodeCpuCpuPingpong"] = p.interNodeCpuCpuPingpong;
  j["interNodeGpuGpuPingpong"] = p.interNodeGpuGpuPingpong;
  j["d2h"] = p.d2h;
  j["h2d"] = p.h2d;
  j["cudaKernelLaunch"] = p.cudaKernelLaunch;
  j["packDevice"] = p.packDevice;
  j["unpackDevice"] = p.unpackDevice;
  j["packHost"] = p.packHost;
  j["unpackHost"] = p.unpackHost;
}
void from_json(const json &j, tempi::system::Performance &p) {
  j.at("intraNodeCpuCpuPingpong").get_to(p.intraNodeCpuCpuPingpong);
  j.at("intraNodeGpuGpuPingpong").get_to(p.intraNodeGpuGpuPingpong);
  j.at("interNodeCpuCpuPingpong").get_to(p.interNodeCpuCpuPingpong);
  j.at("interNodeGpuGpuPingpong").get_to(p.interNodeGpuGpuPingpong);
  j.at("d2h").get_to(p.d2h);
  j.at("h2d").get_to(p.h2d);
  j.at("cudaKernelLaunch").get_to(p.cudaKernelLaunch);
  j.at("packDevice").get_to(p.packDevice);
  j.at("unpackDevice").get_to(p.unpackDevice);
  j.at("packHost").get_to(p.packHost);
  j.at("unpackHost").get_to(p.unpackHost);
}


nonstd::optional<double> Performance::contig_d2h(int64_t bytes) const {
  if (d2h.empty()) {
    return nonstd::nullopt;
  } else {
    return interp_time(d2h, bytes);
  }
}

nonstd::optional<double>
Performance::pack_host_noncontig(int64_t bytes,
                                       int64_t blockLength) const {
  return interp_2d_opt(packHost, bytes, blockLength);
}
nonstd::optional<double>
Performance::unpack_host_noncontig(int64_t bytes,
                                         int64_t blockLength) const {
  return interp_2d_opt(unpackHost, bytes, blockLength);
}

nonstd::optional<double>
Performance::pack_device_noncontig(int64_t bytes,
                                         int64_t blockLength) const {
  return interp_2d_opt(packDevice, bytes, blockLength);
}
nonstd::optional<double>
Performance::unpack_device_noncontig(int64_t bytes,
                                           int64_t blockLength) const {
  return interp_2d_opt(unpackDevice, bytes, blockLength);
}

nonstd::optional<double>
Performance::send_h2h_contig(bool colocated, int64_t bytes) const {
  return interp_time_opt(
      colocated ? intraNodeCpuCpuPingpong : interNodeCpuCpuPingpong, bytes);
}

nonstd::optional<double>
Performance::send_d2d_contig(bool colocated, int64_t bytes) const {
  return interp_time_opt(
      colocated ? intraNodeGpuGpuPingpong : interNodeGpuGpuPingpong, bytes);
}

nonstd::optional<double> Performance::model_oneshot(
    const Performance::SendNonContigNd::Args &args) const {
  const nonstd::optional<double> ph =
      pack_host_noncontig(args.bytes, args.blockLength);
  const nonstd::optional<double> send =
      send_h2h_contig(args.colocated, args.bytes);
  const nonstd::optional<double> uh =
      unpack_host_noncontig(args.bytes, args.blockLength);
  if (!ph || !send || !uh) {
    return nonstd::nullopt;
  } else {
    return *ph + *send + *uh;
  }
}

nonstd::optional<double> Performance::model_device(
    const Performance::SendNonContigNd::Args &args) const {
  const nonstd::optional<double> pd =
      pack_device_noncontig(args.bytes, args.blockLength);
  const nonstd::optional<double> send =
      send_d2d_contig(args.colocated, args.bytes);
  const nonstd::optional<double> ud =
      unpack_device_noncontig(args.bytes, args.blockLength);
  if (!pd) {
    return nonstd::nullopt;
  } else if (!send) {
    return nonstd::nullopt;
  } else if (!ud) {
    return nonstd::nullopt;
  } else {
    return *pd + *send + *ud;
  }
}

bool export_performance(const Performance &sp) {
  json j = sp;
  std::string s = j.dump(2);
  LOG_INFO("ensure cacheDir " << environment::cacheDir);
  std::filesystem::path path(environment::cacheDir);
  std::filesystem::create_directory(path);
  path /= "perf.json";
  LOG_INFO("open " << path);
  std::ofstream file(path);
  if (file.fail()) {
    LOG_INFO("couldn't open " << path);
    return false;
  }
  LOG_INFO("write " << path);
  file << s;
  file.close();

  return true;
}

bool import_performance(Performance &sp) {
  std::filesystem::path path(environment::cacheDir);
  path /= "perf.json";
  LOG_DEBUG("open " << path);
  std::ifstream file(path);
  if (file.fail()) {
    LOG_WARN("couldn't open " << path);
    return false;
  }
  std::stringstream ss;
  ss << file.rdbuf();
  json j = json::parse(ss.str());
  try {
    sp = j;
  } catch (nlohmann::detail::out_of_range &e) {
    LOG_ERROR("error converting json to Performance: "
              << e.what() << " (delete and run bin/measure-performance)");
  }
  return true;
}

double interp_time(const std::vector<IidTime> a, int64_t bytes) {
  auto opt = interp_time_opt(a, bytes);
  if (!opt) {
    return std::numeric_limits<double>::infinity();
  } else {
    return *opt;
  }
}

nonstd::optional<double> interp_time_opt(const std::vector<IidTime> a,
                                         int64_t bytes) {

  if (a.empty()) {
    return nonstd::nullopt;
  }

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

double interp_2d(const std::vector<std::vector<IidTime>> a, int64_t bytes,
                 int64_t stride) {
  auto opt = interp_2d_opt(a, bytes, stride);
  if (!opt) {
    return std::numeric_limits<double>::infinity();
  } else {
    return *opt;
  }
}

nonstd::optional<double>
interp_2d_opt(const std::vector<std::vector<IidTime>> a, int64_t bytes,
              int64_t stride) {
  assert(stride <= 512);
  if (a.empty()) {
    return nonstd::nullopt;
  }

  /*find the surrounding points for which we have measurements,
    as well as indices into the measurement array for the points

    x1,y1 is lower corner, x2,y2 is higher corner

    the x coverage is complete,
     so we only have to handle the case when y1/y2 is not in the array
  */
  uint8_t yi1 = (log2_floor(bytes) - 6) / 2;
  int64_t y1 = 1ull << (yi1 * 2 + 6);
  uint8_t yi2 = (y1 == bytes) ? yi1 : yi1 + 1;
  int64_t y2 = 1ull << (yi2 * 2 + 6);

  // LOG_SPEW("yi1,yi2=" << int(yi1) << "," << int(yi2) << " y1,y2=" << y1 <<
  // "," << y2);

  uint8_t xi1 = log2_floor(stride);
  int64_t x1 = 1ull << xi1;
  uint8_t xi2 = log2_ceil(stride);
  int64_t x2 = 1ull << xi2;

  // clamp bounds in x direction
  // don't do this for y since we just scale the closest value
  if (xi2 >= a[yi2].size()) {
    LOG_DEBUG("clamp x in 2d interpolation");
    xi2 = a[yi2].size() - 1;
  }
  xi1 = std::min(xi2, xi1);

  // LOG_SPEW("xi1,xi2=" << int(xi1) << "," << int(xi2));

  int64_t x = stride;
  int64_t y = bytes;

  // LOG_SPEW("x (stride) =" << x << " y (bytes) =" << y);

  float sf_x;
  if (xi2 == xi1) {
    sf_x = 0.0;
  } else {
    sf_x = float(x - x1) / float(x2 - x1);
  }
  float sf_y;
  if (yi2 == yi1) {
    sf_y = 0.0;
  } else {
    sf_y = float(y - y1) / float(y2 - y1);
  }

  LOG_SPEW("sf_x=" << sf_x << " sf_y=" << sf_y);

  // message is too big, just scale the stride interpolation
  if (yi2 >= a.size()) {
    float base = (1 - sf_x) * a[a.size() - 1][xi1].time +
                 sf_x * a[a.size() - 1][xi2].time;
    float y_max = 1ull << ((a.size() - 1) * 2 + 6);
    // std::cerr << base << "," << y_max << " " << bytes << "\n";
    return base / y_max * bytes;
  } else {
    const float a_y1_x1 = a[yi1][xi1].time;
    const float a_y1_x2 = a[yi1][xi2].time;
    LOG_SPEW("a_y1_x1-x2=" << a_y1_x1 << " " << a_y1_x2);
    float f_x_y1 = (1 - sf_x) * a_y1_x1 + sf_x * a_y1_x2;
    float f_x_y2 = (1 - sf_x) * a[yi2][xi1].time + sf_x * a[yi2][xi2].time;
    float f_x_y = (1 - sf_y) * f_x_y1 + sf_y * f_x_y2;
    LOG_SPEW("f_x_y=" << f_x_y);
    return f_x_y;
  }
}

} // namespace system
} // namespace tempi