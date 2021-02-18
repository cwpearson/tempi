//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

/* performance counters
 */

#include "counters.hpp"

#include "logging.hpp"

namespace counters {

/*extern*/ Allocator deviceAllocator;
/*extern*/ Allocator hostAllocator;
/*extern*/ CUDArt cudart;
/*extern*/ Irecv irecv;
/*extern*/ Isend isend;
/*extern*/ LibCalls libCalls;
/*extern*/ Modeling modeling;
/*extern*/ Pack2d pack2d;
/*extern*/ Pack3d pack3d;
/*extern*/ Recv recv;
/*extern*/ Send send;

void init() {}

void finalize() {
#ifdef TEMPI_ENABLE_COUNTERS

  int size, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  for (int r = 0; r < size; ++r) {
    if (rank == r) {

      LOG_DEBUG("libcalls::Irecv_num:  " << TEMPI_COUNTER(libCalls, IRECV_NUM));
      LOG_DEBUG(
          "libcalls::Irecv_time: " << TEMPI_COUNTER(libCalls, IRECV_TIME));
      LOG_DEBUG("libcalls::Start_num:  " << TEMPI_COUNTER(libCalls, START_NUM));
      LOG_DEBUG(
          "libcalls::Start_time: " << TEMPI_COUNTER(libCalls, START_TIME));
      LOG_DEBUG("libcalls::Send_init_num:  " << TEMPI_COUNTER(libCalls,
                                                              SEND_INIT_NUM));
      LOG_DEBUG("libcalls::Send_init_time: " << TEMPI_COUNTER(libCalls,
                                                              SEND_INIT_TIME));

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

      LOG_DEBUG("send::num_oneshot: " << TEMPI_COUNTER(send, NUM_ONESHOT));
      LOG_DEBUG("send::num_device: " << TEMPI_COUNTER(send, NUM_DEVICE));
      LOG_DEBUG("isend::num_oneshot: " << TEMPI_COUNTER(isend, NUM_ONESHOT));
      LOG_DEBUG("isend::num_device: " << TEMPI_COUNTER(isend, NUM_DEVICE));
      LOG_DEBUG("recv::num_oneshot: " << TEMPI_COUNTER(recv, NUM_ONESHOT));
      LOG_DEBUG("recv::num_device: " << TEMPI_COUNTER(recv, NUM_DEVICE));
      LOG_DEBUG("irecv::num_oneshot: " << TEMPI_COUNTER(irecv, NUM_ONESHOT));
      LOG_DEBUG("irecv::num_device: " << TEMPI_COUNTER(irecv, NUM_DEVICE));

      LOG_DEBUG("pack2d::num_packs:    " << TEMPI_COUNTER(pack2d, NUM_PACKS));
      LOG_DEBUG("pack2d::num_unpacks:  " << TEMPI_COUNTER(pack2d, NUM_UNPACKS));
      LOG_DEBUG("pack3d::num_packs:    " << TEMPI_COUNTER(pack3d, NUM_PACKS));
      LOG_DEBUG("pack3d::num_unpacks:  " << TEMPI_COUNTER(pack3d, NUM_UNPACKS));

      LOG_DEBUG("cudart::launch_time:" << TEMPI_COUNTER(cudart, LAUNCH_TIME));
      LOG_DEBUG("cudart::launch_num:" << TEMPI_COUNTER(cudart, LAUNCH_NUM));
      LOG_DEBUG("cudart::event_record_time:"
                << TEMPI_COUNTER(cudart, EVENT_RECORD_TIME));
      LOG_DEBUG("cudart::event_record_num:" << TEMPI_COUNTER(cudart,
                                                             EVENT_RECORD_NUM));
      LOG_DEBUG("cudart::event_query_time:" << TEMPI_COUNTER(cudart,
                                                             EVENT_QUERY_TIME));
      LOG_DEBUG(
          "cudart::event_query_num:" << TEMPI_COUNTER(cudart, EVENT_QUERY_NUM));
      LOG_DEBUG(
          "cudart::event_sync_time:" << TEMPI_COUNTER(cudart, EVENT_SYNC_TIME));
      LOG_DEBUG(
          "cudart::event_sync_num:" << TEMPI_COUNTER(cudart, EVENT_SYNC_NUM));
      LOG_DEBUG("cudart::stream_sync_time:" << TEMPI_COUNTER(cudart,
                                                             STREAM_SYNC_TIME));
      LOG_DEBUG(
          "cudart::stream_sync_num:" << TEMPI_COUNTER(cudart, STREAM_SYNC_NUM));
      LOG_DEBUG("cudart::memcpy_async_time:"
                << TEMPI_COUNTER(cudart, MEMCPY_ASYNC_TIME));
      LOG_DEBUG("cudart::memcpy_async_num:" << TEMPI_COUNTER(cudart,
                                                             MEMCPY_ASYNC_NUM));
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

#endif
}

} // namespace counters
