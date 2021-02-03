//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "packer.hpp"
#include "symbols.hpp"
#include "type_cache.hpp"

#include <mpi.h>

/* management of async operations */
namespace async {

// block until the managed request is complete.
// if the request is not managed, just MPI_Wait
int wait(MPI_Request *request, MPI_Status *status);

// create a new managed Isend operation using the provided packer and start it
void start_isend(const StridedBlock &sb, Packer &packer, PARAMS_MPI_Isend);

// create a new managed Irecv operation using the provided packer and start it
void start_irecv(const StridedBlock &sb, Packer &packer, PARAMS_MPI_Irecv);

// attempt to progress all active operations.
// returns an MPI error code
int try_progress();

// any operations needed during finalize
void finalize();

}; // namespace async