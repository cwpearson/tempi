#include "async_operation.hpp"

#include "allocators.hpp"
#include "cuda_runtime.hpp"
#include "logging.hpp"
#include "symbols.hpp"

#include <map>
#include <memory>

class AsyncOperation {
public:
  virtual ~AsyncOperation() {}

  // try to make progress on the operation
  virtual int wake() = 0;

  // true if operation will need to be woken up in the future to progress
  virtual bool needs_wake() = 0;

  // block until operation is complete. returns an MPI error code
  virtual int wait(MPI_Status *status) = 0;
};

// active async operations
std::map<MPI_Request *, std::unique_ptr<AsyncOperation>> active;

/*
Manages the state of a particular Isend
CUDA - the CUDA transfer is active or completion is not detected yet
MPI - the MPI_Isend has been issued
*/
class Isend : public AsyncOperation {
  Packer &packer_;
  MPI_Request *request_;

  enum class State { CUDA, MPI };
  State state_;

  void *packedBuf_;
  MPI_Aint packedSize_;

  cudaEvent_t event_;

public:
  Isend(Packer &packer, PARAMS_MPI_Isend)
      : packer_(packer), request_(request), state_(State::CUDA),
        packedBuf_(nullptr), packedSize_(0) {

    // TODO, this could be slow. should probably have a TEMPI cudaEvent_t cache
    CUDA_RUNTIME(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));

    // allocate intermediate space
    MPI_Aint lb, extent;
    MPI_Type_get_extent(datatype, &lb, &extent);
    packedSize_ = extent * count;
    packedBuf_ = hostAllocator.allocate(packedSize_);
    LOG_SPEW("buffer is @" << uintptr_t(packedBuf_));

    // issue pack operation
    int position = 0;
    packer.pack_async(packedBuf_, &position, buf, count, event_);
    LOG_SPEW("Isend():: issued pack");

    // initialize Isend and request
    libmpi.MPI_Send_init(packedBuf_, packedSize_, MPI_PACKED, dest, tag, comm,
                         request_);
    LOG_SPEW("Isend():: init'ed Send");
  }
  ~Isend() {
    hostAllocator.deallocate(packedBuf_, packedSize_);
    assert(event_);
    CUDA_RUNTIME(cudaEventDestroy(event_));
  }

  MPI_Request *request() { return request_; };

  virtual int wake() override {
    switch (state_) {
    case State::CUDA: {
      cudaError_t err = cudaEventQuery(event_);
      if (cudaSuccess == err) {
        state_ = State::MPI;
        return libmpi.MPI_Start(request_);
      } else if (cudaErrorNotReady == err) {
        return MPI_SUCCESS; // still waiting on CUDA
      } else {
        CUDA_RUNTIME(err);
        return MPI_ERR_UNKNOWN;
      }
    }
    case State::MPI: {
      // no next state
      return MPI_SUCCESS;
    }
    default: {
      LOG_FATAL("unexpected state")
    }
    }
  }

  virtual int wait(MPI_Status *status) override {
    while (state_ != State::MPI) {
      wake();
    }
    return libmpi.MPI_Wait(request_, status);
  }

  virtual bool needs_wake() override { return state_ != State::MPI; }
};

/*
Manages the state of a particular Irecv
MPI - the MPI_Irecv has been issued
CUDA - the CUDA transfer is active or completion is not detected yet
*/
class Irecv : public AsyncOperation {
  Packer &packer_;
  MPI_Request *request_;
  void *buf_;
  int count_;

  enum class State { CUDA, MPI };
  State state_;

  void *packedBuf_;
  MPI_Aint packedSize_;

  cudaEvent_t event_;

public:
  Irecv(Packer &packer, PARAMS_MPI_Irecv)
      : packer_(packer), request_(request), buf_(buf), count_(count),
        state_(State::MPI), packedBuf_(nullptr), packedSize_(0) {

    // TODO, this could be slow. should probably have a TEMPI cudaEvent_t cache
    CUDA_RUNTIME(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));

    // allocate intermediate space
    MPI_Aint lb, extent;
    MPI_Type_get_extent(datatype, &lb, &extent);
    packedSize_ = extent * count;
    packedBuf_ = hostAllocator.allocate(packedSize_);

    // issue MPI_Irecv
    libmpi.MPI_Irecv(packedBuf_, packedSize_, MPI_PACKED, source, tag, comm,
                     request_);
    LOG_SPEW("Irecv(): issued Irecv");
  }
  ~Irecv() {
    hostAllocator.deallocate(packedBuf_, packedSize_);
    assert(event_);
    CUDA_RUNTIME(cudaEventDestroy(event_));
  }

  MPI_Request *request() { return request_; };

  virtual int wake() override {
    switch (state_) {
    case State::MPI: {
      // TODO: handle status
      int flag;
      int err = libmpi.MPI_Test(request_, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        // issue unpack operation
        int position = 0;
        packer_.unpack_async(packedBuf_, &position, buf_, count_, event_);
        state_ = State::CUDA;
      } else {
        // wait
      }
      return err;
    }
    case State::CUDA: {
      // no next state
      return MPI_SUCCESS;
    }
    default: {
      LOG_FATAL("unexpected state")
    }
    }
  }

  virtual int wait(MPI_Status *status) override {
    while (state_ != State::MPI) {
      wake();
    }
    return libmpi.MPI_Wait(request_, status);
  }

  virtual bool needs_wake() override { return state_ != State::MPI; }
};

namespace async {

MPI_Request *start_isend(Packer &packer, PARAMS_MPI_Isend) {
  std::unique_ptr<Isend> op = std::make_unique<Isend>(packer, ARGS_MPI_Isend);
  MPI_Request *req = op->request();
  active[req] = std::move(op);
  return req;
}

MPI_Request *start_irecv(Packer &packer, PARAMS_MPI_Irecv) {
  std::unique_ptr<Irecv> op = std::make_unique<Irecv>(packer, ARGS_MPI_Irecv);
  MPI_Request *req = op->request();
  active[req] = std::move(op);
  return req;
}

int wait(MPI_Request *request, MPI_Status *status) {

  auto ii = active.find(request);
  if (active.end() != ii) {
    LOG_SPEW("async::wait() on managed request " << uintptr_t(request));
    int err = ii->second->wait(status);
    active.erase(ii);
    LOG_SPEW("async::wait() cleaned up request "
             << uintptr_t(request) << "(" << active.size() << " remaining)");
    return err;
  } else {
    return libmpi.MPI_Wait(request, status);
  }
}

int try_progress() {
  for (auto &kv : active) {
    if (kv.second->needs_wake()) {
      int err = kv.second->wake();
      if (err != MPI_SUCCESS) {
        return err;
      }
    }
  }
  return MPI_SUCCESS;
}

void finalize() {
  if (!active.empty()) {
    LOG_ERROR("there were "
              << active.size()
              << " managed async operations unterminated at finalize");
  }
}

}; // namespace async