#include "allocators.hpp"

#include <nvToolsExt.h>

/*extern*/ SlabAllocator<char, device_allocator<char>> deviceAllocator;
/*extern*/ SlabAllocator<char, host_allocator<char>> hostAllocator;

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
