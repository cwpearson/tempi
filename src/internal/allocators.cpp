//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "allocators.hpp"

#include <nvToolsExt.h>

/*extern*/ SlabAllocator<char, device_allocator<char>> deviceAllocator(counters::deviceAllocator);
/*extern*/ SlabAllocator<char, host_allocator<char>> hostAllocator(counters::hostAllocator);

namespace allocators {

void init() {
  nvtxRangePush("allocators::init");
  nvtxRangePop();
}

void finalize() {
  nvtxRangePush("allocators::finalize");

  deviceAllocator.release_all();
  hostAllocator.release_all();

  nvtxRangePop();
}

void release_all() {
  deviceAllocator.release_all();
  hostAllocator.release_all();
}

} // namespace allocators
