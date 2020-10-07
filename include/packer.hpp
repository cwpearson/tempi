#pragma once

class Packer {
public:
  virtual ~Packer() {}

  virtual void pack(void *outbuf,  // write here
                    int *position, // current position in outbuf (bytes),
                    const void *inbuf,
                    const int incount // [in] number of input data items
  ) const = 0;
};

class PackerStride2 : public Packer {
  unsigned blockLength_;
  unsigned count_[2];
  unsigned stride_[2]; // 0 is inner stride, 1 is outer stride

public:
  PackerStride2(unsigned blockLength, unsigned count0, unsigned stride0, unsigned count1, unsigned stride1);
  void pack(void *outbuf, int *position, const void *inbuf,
            const int incount) const override;
};