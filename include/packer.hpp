#pragma once

#include "dim3.hpp"

class Packer {
public:
  virtual ~Packer() {}

  virtual void pack(void *outbuf,  // write here
                    int *position, // current position in outbuf (bytes),
                    const void *inbuf,
                    const int incount // [in] number of input data items
  ) const = 0;
};