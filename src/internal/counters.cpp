//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

/* performance counters
 */

#include "counters.hpp"

#include "logging.hpp"

namespace counters {

/*extern*/ std::unordered_map<counters, double> modeling;
/*extern*/ std::unordered_map<counters, double> pack3d;

void init() {}

void finalize() {
  LOG_INFO("modeling::cache_miss: " << modeling[CACHE_MISS]);
  LOG_INFO("modeling::cache_hit:  " << modeling[CACHE_HIT]);
  LOG_INFO("modeling::wall_time:  " << modeling[WALL_TIME]);
  LOG_INFO("pack3d::num_packs:    " << pack3d[NUM_PACKS]);
  LOG_INFO("pack3d::num_unpacks:  " << pack3d[NUM_UNPACKS]);
}

} // namespace counters