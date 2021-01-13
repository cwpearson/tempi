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

void to_json(json &j, const IidTime &p) {
  j["time"] = p.time;
  j["iid"] = p.iid;
}
void from_json(const json &j, IidTime &p) {
  j.at("time").get_to(p.time);
  j.at("iid").get_to(p.iid);
}

// === SystemPerformance ===
void to_json(json &j, const SystemPerformance &p) {
  j["intraNodeCpuCpuPingpong"] = p.intraNodeCpuCpuPingpong;
  j["intraNodeGpuGpuPingpong"] = p.intraNodeGpuGpuPingpong;
  j["interNodeCpuCpuPingpong"] = p.interNodeCpuCpuPingpong;
  j["interNodeGpuGpuPingpong"] = p.interNodeGpuGpuPingpong;
  j["d2h"] = p.d2h;
  j["h2d"] = p.h2d;
  j["cudaKernelLaunch"] = p.cudaKernelLaunch;
  j["packDevice"] = p.packDevice;
  j["packHost"] = p.packHost;
}
void from_json(const json &j, SystemPerformance &p) {
  j.at("intraNodeCpuCpuPingpong").get_to(p.intraNodeCpuCpuPingpong);
  j.at("intraNodeGpuGpuPingpong").get_to(p.intraNodeGpuGpuPingpong);
  j.at("interNodeCpuCpuPingpong").get_to(p.interNodeCpuCpuPingpong);
  j.at("interNodeGpuGpuPingpong").get_to(p.interNodeGpuGpuPingpong);
  j.at("d2h").get_to(p.d2h);
  j.at("h2d").get_to(p.h2d);
  j.at("cudaKernelLaunch").get_to(p.cudaKernelLaunch);
  j.at("packDevice").get_to(p.packDevice);
  j.at("packHost").get_to(p.packHost);
}

bool export_system_performance(const SystemPerformance &sp) {
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

bool import_system_performance(SystemPerformance &sp) {
  std::filesystem::path path(environment::cacheDir);
  path /= "perf.json";
  LOG_DEBUG("open " << path);
  std::ifstream file(path);
  if (file.fail()) {
    LOG_INFO("couldn't open " << path);
    return false;
  }
  std::stringstream ss;
  ss << file.rdbuf();
  json j = json::parse(ss.str());
  try {
    sp = j;
  } catch (nlohmann::detail::out_of_range &e) {
    LOG_ERROR("error converting json to SystemPerformance: "
              << e.what() << " (delete and run bin/measure-performance)");
  }
  return true;
}
