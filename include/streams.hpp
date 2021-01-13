//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cuda_runtime.h>

extern cudaStream_t commStream;
extern cudaStream_t kernStream;

void streams_init();
void streams_finalize();