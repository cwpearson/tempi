//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

/* performance counters
 */

#include "counters.hpp"

#include "logging.hpp"

namespace counters {

/*extern*/ Modeling modeling;
/*extern*/ Pack2d pack2d;
/*extern*/ Pack3d pack3d;
/*extern*/ Allocator deviceAllocator;
/*extern*/ Allocator hostAllocator;

void init() {}

void finalize() {
#ifdef TEMPI_ENABLE_COUNTERS

  int size, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for (int r = 0; r < size; ++r) {
    if (rank == r) {
      LOG_DEBUG("host_allocator::num_allocs: " << TEMPI_COUNTER(hostAllocator,
                                                                NUM_ALLOCS));
      LOG_DEBUG("host_allocator::num_deallocs: "
                << TEMPI_COUNTER(hostAllocator, NUM_DEALLOCS));
      LOG_DEBUG("host_allocator::num_requests: "
                << TEMPI_COUNTER(hostAllocator, NUM_REQUESTS));
      LOG_DEBUG("host_allocator::num_releases: "
                << TEMPI_COUNTER(hostAllocator, NUM_RELEASES));
      LOG_DEBUG("host_allocator::max_usage: " << TEMPI_COUNTER(hostAllocator,
                                                               MAX_USAGE));

      LOG_DEBUG("device_allocator::num_allocs: "
                << TEMPI_COUNTER(deviceAllocator, NUM_ALLOCS));
      LOG_DEBUG("device_allocator::num_deallocs: "
                << TEMPI_COUNTER(deviceAllocator, NUM_DEALLOCS));
      LOG_DEBUG("device_allocator::num_requests: "
                << TEMPI_COUNTER(deviceAllocator, NUM_REQUESTS));
      LOG_DEBUG("device_allocator::num_releases: "
                << TEMPI_COUNTER(deviceAllocator, NUM_RELEASES));
      LOG_DEBUG("device_allocator::max_usage: "
                << TEMPI_COUNTER(deviceAllocator, MAX_USAGE));

      LOG_DEBUG(
          "modeling::cache_miss: " << TEMPI_COUNTER(modeling, CACHE_MISS));
      LOG_DEBUG("modeling::cache_hit:  " << TEMPI_COUNTER(modeling, CACHE_HIT));
      LOG_DEBUG("modeling::wall_time:  " << TEMPI_COUNTER(modeling, WALL_TIME));
      LOG_DEBUG("pack2d::num_packs:    " << TEMPI_COUNTER(pack2d, NUM_PACKS));
      LOG_DEBUG("pack2d::num_unpacks:  " << TEMPI_COUNTER(pack2d, NUM_UNPACKS));
      LOG_DEBUG("pack3d::num_packs:    " << TEMPI_COUNTER(pack3d, NUM_PACKS));
      LOG_DEBUG("pack3d::num_unpacks:  " << TEMPI_COUNTER(pack3d, NUM_UNPACKS));
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

#endif
}

} // namespace counters