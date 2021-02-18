//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#define TEMPI_ENABLE_COUNTERS

#include <cstdint>

namespace counters {

struct Allocator {
  uint64_t NUM_REQUESTS;
  uint64_t NUM_RELEASES;
  uint64_t NUM_ALLOCS;
  uint64_t NUM_DEALLOCS;
  uint64_t CURRENT_USAGE;
  uint64_t MAX_USAGE;
};

struct CUDArt {
  double LAUNCH_TIME;
  double EVENT_RECORD_TIME;
  double EVENT_QUERY_TIME;
  double EVENT_SYNC_TIME;
  double STREAM_SYNC_TIME;
  double MEMCPY_ASYNC_TIME;
  unsigned LAUNCH_NUM;
  unsigned EVENT_RECORD_NUM;
  unsigned EVENT_QUERY_NUM;
  unsigned EVENT_SYNC_NUM;
  unsigned STREAM_SYNC_NUM;
  unsigned MEMCPY_ASYNC_NUM;
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

struct Send {
  unsigned NUM_ONESHOT;
  unsigned NUM_DEVICE;
};
struct Recv {
  unsigned NUM_ONESHOT;
  unsigned NUM_DEVICE;
};
struct Isend {
  unsigned NUM_ONESHOT;
  unsigned NUM_DEVICE;
};
struct Irecv {
  unsigned NUM_ONESHOT;
  unsigned NUM_DEVICE;
};

/* underlying MPI library calls
 */
struct LibCalls {
  unsigned IRECV_NUM;
  double IRECV_TIME;
  unsigned ISEND_NUM;
  double ISEND_TIME;
  unsigned START_NUM;
  double START_TIME;
  unsigned SEND_INIT_NUM;
  double SEND_INIT_TIME;
};

extern Allocator deviceAllocator;
extern Allocator hostAllocator;
extern CUDArt cudart;
extern LibCalls libCalls;
extern Modeling modeling;
extern Pack2d pack2d;
extern Pack3d pack3d;
extern Send send;
extern Recv recv;
extern Isend isend;
extern Irecv irecv;

void init();
void finalize();

} // namespace counters

#ifdef TEMPI_ENABLE_COUNTERS
#define TEMPI_COUNTER_EXPR(expr) expr
#define TEMPI_COUNTER(group, key) counters::group.key
#define TEMPI_COUNTER_OP(group, key, op) (counters::group.key) op
#else
#define TEMPI_COUNTER_EXPR(expr)
#define TEMPI_COUNTER(group, key)
#define TEMPI_COUNTER_OP(group, key, op)
#endif