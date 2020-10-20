#include "allocators.hpp"

#include <nvToolsExt.h>

/*extern*/ SlabAllocator<char, device_allocator<char>> deviceAllocator;
/*extern*/ SlabAllocator<char, host_allocator<char>> hostAllocator;

void allocators_init() {
  nvtxRangePush("allocators_init");
  nvtxRangePop();
}

void allocators_finalize() {
  nvtxRangePush("allocators_finalize");

  deviceAllocator.release_all();
  hostAllocator.release_all();

  nvtxRangePop();
}