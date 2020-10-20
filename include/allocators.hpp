#pragma once

#include "allocator_device.hpp"
#include "allocator_host.hpp"
#include "allocator_slab.hpp"

void allocators_init();
void allocators_finalize();

extern SlabAllocator<char, device_allocator<char>> deviceAllocator;
extern SlabAllocator<char, host_allocator<char>> hostAllocator;