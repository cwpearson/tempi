#include "task_staged_isend.hpp"

#include "cuda_runtime.hpp"
#include "logging.hpp"
#include "requests.hpp"
#include "streams.hpp"
#include "symbols.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <vector>

StagedIsend::StagedIsend(const void *buf, int count, MPI_Datatype datatype,
                         int dest, int tag, MPI_Comm comm, MPI_Request *request)
    : state_(State::INIT), buf_(buf), count_(count), datatype_(datatype),
      dest_(dest), tag_(tag), comm_(comm), request_(request) {
  // create a host buffer
  {
    int size;
    MPI_Type_size(datatype_, &size);
    uint64_t usize = uint64_t(size) * count_;
    hostBuf_.resize(usize);
  }

  // figure out device
  {
    cudaPointerAttributes attr = {};
    CUDA_RUNTIME(cudaPointerGetAttributes(&attr, buf));
    device_ = attr.device;
  }

  // create event
  CUDA_RUNTIME(cudaSetDevice(device_));
  CUDA_RUNTIME(cudaEventCreate(&event_));
}

StagedIsend::~StagedIsend() { CUDA_RUNTIME(cudaEventDestroy(event_)); }

bool StagedIsend::ready() {
  switch (state_) {
  case State::INIT: {
    return true;
  }
  case State::D2H: {
    cudaError_t err = cudaEventQuery(event_);
    if (cudaSuccess == err) {
      return true;
    } else if (cudaErrorNotReady == err) {
      return false;
    } else {
      CUDA_RUNTIME(err);
      LOG_FATAL("unexpected cuda error");
      return false;
    }
  }
  case State::H2H: {
    int flag = false;
    MPI_Test(request_, &flag, MPI_STATUS_IGNORE);
    return flag;
  }
  case State::DONE: { // always ready to be in init
    return true;
  }
  default:
    LOG_FATAL("unexpected state");
  }
}

void StagedIsend::progress() {

  switch (state_) {
  case State::INIT: {
    LOG_SPEW("StagedIsend: INIT -> D2H");
    state_ = State::D2H;
    CUDA_RUNTIME(cudaMemcpyAsync(hostBuf_.data(), buf_, hostBuf_.size(),
                                 cudaMemcpyDefault, commStream));
    CUDA_RUNTIME(cudaEventRecord(event_, commStream));
    LOG_SPEW("StagedIsend: issused D2H copy");
    break;
  }
  case State::D2H: {
    LOG_SPEW("StagedIsend: D2H -> H2H");
    state_ = State::H2H;
    libmpi.MPI_Isend(hostBuf_.data(), count_, datatype_, dest_, tag_, comm_, request_);
    LOG_SPEW("StagedIsend: post Isend");
    release_request(request_);
    LOG_SPEW("StagedIsend: release request");
    break;
  }
  case State::H2H: {
    state_ = State::DONE;
    break;
  }
  case State::DONE: {
    state_ = State::INIT;
    break;
  }
  default:
    LOG_FATAL("unexpected state");
  }
}

bool StagedIsend::done() { return State::DONE == state_; }