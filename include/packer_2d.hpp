//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "packer.hpp"
#include "pack_config.hpp"

#define USE_NEW_PACKER

class Packer2D : public Packer {
  using PackFn = void (*)(void *__restrict__ outbuf,
                          const void *__restrict__ inbuf,
                          const unsigned incount, const unsigned count0,
                          const unsigned count1, const unsigned stride1,
                          const uint64_t extent);

  using UnpackFn = void (*)(void *__restrict__ outbuf,
                            const void *__restrict__ inbuf, const int outcount,
                            unsigned count0, unsigned count1, unsigned stride1,
                            const uint64_t extent);

  unsigned offset_;
  unsigned blockLength_;
  unsigned count_;
  unsigned stride_;
  unsigned extent_;

  Pack2DConfig config_;

public:
  Packer2D(unsigned off, unsigned blockLength, unsigned count, unsigned stride,
           unsigned extent);
  void pack(void *outbuf, int *position, const void *inbuf,
            const int incount) const override;
  void pack_async(void *outbuf, int *position, const void *inbuf,
                  const int incount, cudaEvent_t event = 0) const override;
  void unpack(const void *inbuf, int *position, void *outbuf,
              const int outcount) const override;
  void unpack_async(const void *inbuf, int *position, void *outbuf,
                    const int outcount, cudaEvent_t event = 0) const override;

  /*public so can be used in benchmarking. If event is not null, record kernel
   * time in event*/
  void launch_pack(void *outbuf, int *position, const void *inbuf,
                   const int incount, cudaStream_t stream,
                   cudaEvent_t kernelStart = {},
                   cudaEvent_t kernelStop = {}) const;
  void launch_unpack(const void *inbuf, int *position, void *outbuf,
                     const int outcount, cudaStream_t stream,
                     cudaEvent_t kernelStart = {},
                     cudaEvent_t kernelStop = {}) const;

private:
};