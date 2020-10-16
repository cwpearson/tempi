#pragma once

#include <mpi.h>

// future calls to MPI_Wait will block until release_request is called
void block_request(MPI_Request *request);
void release_request(MPI_Request *request);
// block until release_request is called
void wait(MPI_Request *request);