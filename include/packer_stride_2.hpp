#pragma once

#include "packer.hpp"

class PackerStride2 : public Packer {
  unsigned blockLength_;
  unsigned count_[2];
  unsigned stride_[2]; // 0 is inner stride, 1 is outer stride

  int wordSize_; // number of bytes each thread will load
  Dim3 gd_, bd_; // grid dim and block dim for pack kernel

public:
  PackerStride2(unsigned blockLength, unsigned count0, unsigned stride0,
                unsigned count1, unsigned stride1);
  void pack(void *outbuf, int *position, const void *inbuf,
            const int incount) const override;
};