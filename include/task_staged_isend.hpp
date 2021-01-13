//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "task_staged_isend.hpp"

#include "worker.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <vector>

class StagedIsend : public WorkerTask {

  enum class State {
    INIT, // haven't started yet
    D2H,  // copying from GPU to host
    H2H,  // copying from host to host
    DONE  // done
  };

  State state_;

  // Isend params
  const void *buf_;
  int count_;
  MPI_Datatype datatype_;
  int dest_;
  int tag_;
  MPI_Comm comm_;
  MPI_Request *request_;

  std::vector<unsigned char>
      hostBuf_; // temporary host buffer, also tracks size

  int device_;
  cudaEvent_t event_;

public:
  StagedIsend(const void *buf, int count, MPI_Datatype datatype,
                         int dest, int tag, MPI_Comm comm, MPI_Request *request);
  ~StagedIsend();

  bool ready() override;
  void progress() override;
  bool done() override;
};

