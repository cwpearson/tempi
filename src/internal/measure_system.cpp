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

// === Bandwidth ===
void to_json(json &j, const Bandwidth &p) {
  j["time"] = p.time;
  j["bytes"] = p.bytes;
  j["iid"] = p.iid;
}
void from_json(const json &j, Bandwidth &p) {
  j.at("time").get_to(p.time);
  j.at("bytes").get_to(p.bytes);
  j.at("iid").get_to(p.iid);
}

// === SystemPerformance ===
void to_json(json &j, const SystemPerformance &p) {
  j["pingpong"] = p.pingpong;
  j["d2h"] = p.d2h;
  j["h2d"] = p.h2d;
  j["cudaKernelLaunch"] = p.cudaKernelLaunch;
}
void from_json(const json &j, SystemPerformance &p) {
  j.at("pingpong").get_to(p.pingpong);
  j.at("d2h").get_to(p.d2h);
  j.at("h2d").get_to(p.h2d);
  j.at("cudaKernelLaunch").get_to(p.cudaKernelLaunch);
}

bool export_system_performance(const SystemPerformance &sp) {
  json j = sp;
  std::string s = j.dump(2);
  LOG_INFO("ceated json:" << s);

  LOG_INFO("ensure cacheDir " << environment::cacheDir);
  std::filesystem::path path(environment::cacheDir);
  std::filesystem::create_directory(path);
  path /= "perf.json";
  LOG_INFO("write " << path);
  std::ofstream file(path);
  file << s;
  file.close();
  return true;
}

bool import_system_performance(SystemPerformance &sp) {

  std::filesystem::path path(environment::cacheDir);
  path /= "perf.json";
  std::ifstream file(path);
  if (file.bad()) {
    LOG_ERROR("error opening " << path);
    return false;
  }
  std::stringstream ss;
  ss << file.rdbuf();
  LOG_INFO(ss.str());
  json j = json::parse(ss.str());
  sp = j;
  return true;
}
