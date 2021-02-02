//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

class Pack2DConfig {
  using PackFn = void (*)(void *__restrict__ outbuf,
                          const void *__restrict__ inbuf,
                          const unsigned incount, const unsigned count0,
                          const unsigned count1, const unsigned stride1,
                          const uint64_t extent);

  using UnpackFn = void (*)(void *__restrict__ outbuf,
                            const void *__restrict__ inbuf, const int outcount,
                            unsigned count0, unsigned count1, unsigned stride1,
                            const uint64_t extent);

  dim3 gd_, bd_;

public:
  PackFn packfn;
  UnpackFn unpackfn;

  //impl in pack_kernels
  Pack2DConfig(unsigned offset, unsigned blockLength, unsigned blockCount);

  dim3 dim_grid(int count) const {
    dim3 gd = gd_;
    gd.z = count;
    return gd;
  }
  const dim3 &dim_block() const {return bd_;}
};
