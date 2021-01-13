//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <mpi.h>

// future calls to MPI_Wait will block until release_request is called
void block_request(MPI_Request *request);
void release_request(MPI_Request *request);
// block until release_request is called
void wait(MPI_Request *request);