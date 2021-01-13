//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "packer.hpp"
#include "symbols.hpp"

namespace send {

/* pack data into GPU buffer and send */
int pack_device_send(int device, Packer &packer, PARAMS_MPI_Send);

/* pack data into pinned buffer and send */
int pack_host_send(int device, Packer &packer, PARAMS_MPI_Send);

/* pack data into GPU buffer, copy to pinned, and send */
int pack_device_stage(int device, Packer &packer, PARAMS_MPI_Send);

/* copy to pinned, and send */
int staged(int numBytes, PARAMS_MPI_Send);

int impl(PARAMS_MPI_Send);

} // namespace send

