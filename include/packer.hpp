#pragma once

#include "dim3.hpp"

#include <cuda_runtime.h>

/*
 */
class Packer {
public:
  virtual ~Packer() {}

  // get the stream and device info for an operation
  struct LaunchInfo {
    cudaStream_t stream;
  };
  static LaunchInfo pack_launch_info(const void *inbuf);
  static LaunchInfo unpack_launch_info(const void *outbuf);

  // launch unpack operation into the TEMPI stream for outbuf
  virtual void pack_async(void *outbuf,  // write here
                          int *position, // current position in outbuf (bytes),
                          const void *inbuf,
                          const int incount // [in] number of input data items
  ) const = 0;

  // return once object is packed into outbuf
  virtual void pack(void *outbuf,  // write here
                    int *position, // current position in outbuf (bytes),
                    const void *inbuf,
                    const int incount // [in] number of input data items
  ) const = 0;

  // return once packed object is unpacked into outbuf
  virtual void unpack(const void *inbuf, int *position, void *outbuf,
                      const int outcount) const = 0;
};