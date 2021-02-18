//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "packer.hpp"

#include "cuda_runtime.hpp"
#include "logging.hpp"
#include "streams.hpp"

Packer::LaunchInfo Packer::pack_launch_info(const void *inbuf) {
  (void)inbuf;
  Packer::LaunchInfo ret{.stream = kernStream};
  return ret;
}

Packer::LaunchInfo Packer::unpack_launch_info(const void *outbuf) {
  (void)outbuf;
  Packer::LaunchInfo ret{.stream = kernStream};
  return ret;
}
