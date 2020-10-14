#pragma once

#include "packer.hpp"

class PackerStride1 : public Packer {
  unsigned blockLength_;
  unsigned count_;
  unsigned stride_;

  int wordSize_; // number of bytes each thread will load
  Dim3 gd_, bd_; // grid dim and block dim for pack kernel

public:
  PackerStride1(unsigned blockLength, unsigned count, unsigned stride);
  void pack(void *outbuf, int *position, const void *inbuf,
            const int incount) const override;
  void unpack(const void *inbuf, int *position, void *outbuf,
              const int outcount) const override;
};