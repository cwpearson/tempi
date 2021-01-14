//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cuda_runtime.h>

namespace events {
void init();
void finalize();

// get an event from the pool
cudaEvent_t request();
void release(cudaEvent_t event);
} // namespace events
