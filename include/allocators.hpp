#pragma once

#include "allocator_device.hpp"
#include "allocator_host.hpp"
#include "allocator_slab.hpp"

namespace allocators {
void init();
void finalize();
// release any cached memory (may slow future allocator performance)
void release_all();
} // namespace allocators

extern SlabAllocator<char, device_allocator<char>> deviceAllocator;
extern SlabAllocator<char, host_allocator<char>> hostAllocator;