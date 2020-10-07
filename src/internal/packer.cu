#include "packer.hpp"

#include "streams.hpp"

PackerStride2::PackerStride2(unsigned blockLength, unsigned count0,
                             unsigned stride0, unsigned count1,
                             unsigned stride1) {
  blockLength_ = blockLength;

  count_[0] = count0;
  count_[1] = count1;

  stride_[0] = stride0;
  stride_[1] = stride1;
}

void PackerStride2::pack(void *outbuf, int *position, const void *inbuf,
                         const int incount) const {

  char *__restrict__ op = reinterpret_cast<char *>(outbuf);
  const char *__restrict__ ip = reinterpret_cast<const char *>(inbuf);

  for (int i = 0; i < incount; ++i) {
    char *__restrict__ dst =
        op + *position + i * count_[1] * count_[0] * blockLength_;
    const char *__restrict__ src =
        ip + i * stride_[1] * count_[1] * stride_[0] * count_[0];

    for (unsigned z = 0; z < count_[1]; ++z) {
      for (unsigned y = 0; y < count_[0]; ++y) {
        for (unsigned x = 0; x < blockLength_; ++x) {
          dst[z * count_[0] * blockLength_ + y * blockLength_ + x] =
              src[z * stride_[1] + y * stride_[0] + x];
        }
      }
    }
  }
  (*position) += incount * count_[1] * count_[0] * blockLength_;
}