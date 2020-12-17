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

