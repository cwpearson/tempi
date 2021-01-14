//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "packer.hpp"

class Packer1D : public Packer {
  unsigned offset_;
  unsigned extent_;

public:
  Packer1D(unsigned off, unsigned extent);
  void pack(void *outbuf, int *position, const void *inbuf,
            const int incount) const override;
  void pack_async(void *outbuf, int *position, const void *inbuf,
                  const int incount, cudaEvent_t event = 0) const override;
  void unpack(const void *inbuf, int *position, void *outbuf,
              const int outcount) const override;
  void unpack_async(const void *inbuf, int *position, void *outbuf,
                    const int outcount, cudaEvent_t event = 0) const override;

private:
  void launch_pack(void *outbuf, int *position, const void *inbuf,
                   const int incount, cudaStream_t stream) const;
  void launch_unpack(const void *inbuf, int *position, void *outbuf,
                     const int outcount, cudaStream_t stream) const;
};