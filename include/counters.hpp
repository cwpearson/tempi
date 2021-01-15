//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <unordered_map>

// #define TEMPI_ENABLE_COUNTERS

namespace counters {

enum class Key { CACHE_MISS, CACHE_HIT, WALL_TIME, NUM_PACKS, NUM_UNPACKS };

extern std::unordered_map<Key, double> modeling;
extern std::unordered_map<Key, double> pack3d;

void init();
void finalize();


} // namespace counters


#ifdef TEMPI_ENABLE_COUNTERS
#define TEMPI_COUNTER(group, key) counters::group[counters::Key::key]
#define TEMPI_COUNTER_OP(group, key, op) counters::group[counters::Key::key]op
#else
#define TEMPI_COUNTER(group, key)
#define TEMPI_COUNTER_OP(group, key, op)
#endif