//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cuda_runtime.hpp>

/* Compute launch parameters for pack_bytes_warp
 */
class PackConfig {
  using PackFn = void (*)(void *__restrict__ outbuf,
                          const void *__restrict__ inbuf,
                          const unsigned incount, const unsigned count0,
                          const unsigned count1, const unsigned stride1);

  dim3 dimGrid;
  dim3 dimBlock;

public:
  PackFn packfn;
  PackConfig(unsigned blockLength, unsigned blockCount);

  dim3 dim_grid(int count) const;
  dim3 dim_block() const;
};