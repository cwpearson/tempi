//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

/* performance counters
 */

#include "counters.hpp"

#include "logging.hpp"

namespace counters {

/*extern*/ std::unordered_map<Key, double> modeling;
/*extern*/ std::unordered_map<Key, double> pack3d;

void init() {}

void finalize() {
#ifdef TEMPI_ENABLE_COUNTERS
  LOG_DEBUG("modeling::cache_miss: " << TEMPI_COUNTER(modeling, CACHE_MISS));
  LOG_DEBUG("modeling::cache_hit:  " << TEMPI_COUNTER(modeling, CACHE_HIT));
  LOG_DEBUG("modeling::wall_time:  " << TEMPI_COUNTER(modeling, WALL_TIME));
  LOG_DEBUG("pack3d::num_packs:    " << TEMPI_COUNTER(pack3d, NUM_PACKS));
  LOG_DEBUG("pack3d::num_unpacks:  " << TEMPI_COUNTER(pack3d, NUM_UNPACKS));
#endif
}

} // namespace counters
