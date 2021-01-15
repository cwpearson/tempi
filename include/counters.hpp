//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <unordered_map>

namespace counters {

enum counters { CACHE_MISS, CACHE_HIT, WALL_TIME, NUM_PACKS, NUM_UNPACKS };

extern std::unordered_map<counters, double> modeling;
extern std::unordered_map<counters, double> pack3d;

void init();
void finalize();

} // namespace counters