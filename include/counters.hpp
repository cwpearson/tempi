//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#define TEMPI_ENABLE_COUNTERS

#include <cstdint>

namespace counters {

struct Allocator {
    uint64_t numRequests;
    uint64_t numReleases;
    uint64_t NUM_ALLOCS;
    uint64_t NUM_DEALLOCS;
    uint64_t currUsage;
    uint64_t maxUsage;
};

struct Modeling {
    unsigned CACHE_MISS;
    unsigned CACHE_HIT;
    double WALL_TIME;
};

struct Pack2d {
    unsigned NUM_PACKS;
    unsigned NUM_UNPACKS;
};

struct Pack3d {
    unsigned NUM_PACKS;
    unsigned NUM_UNPACKS;
};

extern Modeling modeling;
extern Pack2d pack2d;
extern Pack3d pack3d;
extern Allocator deviceAllocator;
extern Allocator hostAllocator;

void init();
void finalize();


} // namespace counters


#ifdef TEMPI_ENABLE_COUNTERS
#define TEMPI_COUNTER(group, key) counters::group.key
#define TEMPI_COUNTER_OP(group, key, op) (counters::group.key)op
#else
#define TEMPI_COUNTER(group, key)
#define TEMPI_COUNTER_OP(group, key, op)
#endif