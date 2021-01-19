//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "pack_config.hpp"
#include "packer.hpp"

#define USE_NEW_PACKER

class Packer2D : public Packer {
  unsigned offset_;
  unsigned blockLength_;
  unsigned count_;
  unsigned stride_;
#ifdef USE_NEW_PACKER
  PackConfig params_;
#endif

  int wordSize_; // number of bytes each thread will load
  Dim3 gd_, bd_; // grid dim and block dim for pack kernel

public:
  Packer2D(unsigned off, unsigned blockLength, unsigned count, unsigned stride);
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