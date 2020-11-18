#pragma once

#include "dim3.hpp"

#include <cuda_runtime.h>

/*
 */
class Packer {
public:
  virtual ~Packer() {}

#if 0
  // launch unpack operation into the device kernel stream associated with
  // outbuf
  virtual void pack_async(void *outbuf,  // write here
                          int *position, // current position in outbuf (bytes),
                          const void *inbuf,
                          const int incount // [in] number of input data items
  ) const = 0;

  // block until a previous pack_async operation on `inbuf` is done
  void sync(const void *inbuf);
#endif

  // return once object is packed into outbuf
  virtual void pack(void *outbuf,  // write here
                    int *position, // current position in outbuf (bytes),
                    const void *inbuf,
                    const int incount // [in] number of input data items
  ) const = 0;

  // return once packed object is unpacked into outbuf
  virtual void unpack(const void *inbuf, int *position, void *outbuf,
                      const int outcount) const = 0;

protected:
  // get the stream and device info for an operation
  struct LaunchInfo {
    cudaStream_t stream;
    int device;
  };

  static LaunchInfo pack_launch_info(const void *inbuf);
  static LaunchInfo unpack_launch_info(const void *outbuf);
};