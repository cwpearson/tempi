#include "async_operation.hpp"

#include "allocators.hpp"
#include "cuda_runtime.hpp"
#include "events.hpp"
#include "logging.hpp"
#include "symbols.hpp"

#include <map>
#include <memory>

// #define USE_NVTX
#ifdef USE_NVTX
#include <nvToolsExt.h>
#define NVTX_MARK(x) nvtxMark(x)
#define NVTX_RANGE_PUSH(x) nvtxRangePush(x)
#define NVTX_RANGE_POP() nvtxRangePop()
#else
#define NVTX_MARK(x)
#define NVTX_RANGE_PUSH(x)
#define NVTX_RANGE_POP()
#endif

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
std::map<MPI_Request, std::unique_ptr<AsyncOperation>> active;

/*
Manages the state of a particular Isend
CUDA - the CUDA transfer is active or completion is not detected yet
MPI - the MPI_Isend has been issued

It is not safe to store the request pointer
to manipulate the request, as the MPI_Isend caller may move the request
after the call (for example, the request is on the stack).

Isend makes a copy of the MPI_Request.
When MPI completes a request, we'll track the original value.
when the caller provides us with a request that MPI has already completed,
we can be confident we know which operation they intended.

TODO:
It is possible that MPI would reuse a request that's been freed.
Then we could have an old request for an operation the caller has not waited on,
And a new identical MPI request. Then the caller could wait on the old one.
In that case we might have to maintain an ordering of in-flight operations that
use the same MPI_Request and complete them in order.
*/
class Isend : public AsyncOperation {
  Packer &packer_;
  MPI_Request *request_;

  enum class State { CUDA, MPI };
  State state_;

  void *packedBuf_;
  int packedSize_;

  cudaEvent_t event_;

public:
  Isend(Packer &packer, PARAMS_MPI_Isend)
      : packer_(packer), request_(request), state_(State::CUDA),
        packedBuf_(nullptr), packedSize_(0) {
    NVTX_MARK("Isend()");
    event_ = events::request();

    // allocate intermediate space
    MPI_Pack_size(count, datatype, comm, &packedSize_);
    packedBuf_ = hostAllocator.allocate(packedSize_);
    LOG_SPEW("buffer is @" << uintptr_t(packedBuf_));

    // issue pack operation
    int position = 0;
    packer.pack_async(packedBuf_, &position, buf, count, event_);
    LOG_SPEW("Isend():: issued pack");

    // initialize Isend and request
    libmpi.MPI_Send_init(packedBuf_, packedSize_, MPI_PACKED, dest, tag, comm,
                         request_);
    LOG_SPEW("Isend():: init'ed Send, req=" << uintptr_t(*request_));
  }
  ~Isend() {
    hostAllocator.deallocate(packedBuf_, packedSize_);
    events::release(event_);
  }

  virtual int wake() override {
    NVTX_MARK("Isend::wake()");
    switch (state_) {
    case State::CUDA: {
      cudaError_t err = cudaEventQuery(event_);
      if (cudaSuccess == err) {
        NVTX_MARK("Isend:: CUDA->MPI");
        LOG_SPEW("Isend::wake() MPI_Start, req=" << uintptr_t(*request_));
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
  int packedSize_;

  cudaEvent_t event_;

public:
  Irecv(Packer &packer, PARAMS_MPI_Irecv)
      : packer_(packer), request_(request), buf_(buf), count_(count),
        state_(State::MPI), packedBuf_(nullptr), packedSize_(0) {
    NVTX_MARK("Irecv()");
    event_ = events::request();

    // allocate intermediate space
    MPI_Pack_size(count, datatype, comm, &packedSize_);
    packedBuf_ = hostAllocator.allocate(packedSize_);

    NVTX_MARK("Irecv() issue MPI_Irecv");
    // issue MPI_Irecv
    libmpi.MPI_Irecv(packedBuf_, packedSize_, MPI_PACKED, source, tag, comm,
                     request_);
    LOG_SPEW("Irecv(): issued Irecv, req=" << uintptr_t(*request_));
  }
  ~Irecv() {
    hostAllocator.deallocate(packedBuf_, packedSize_);
    assert(event_);
    events::release(event_);
  }

  virtual int wake() override {
    NVTX_MARK("Irecv::wake()");
    switch (state_) {
    case State::MPI: {
      // TODO: handle status
      int flag;
      // FIXME: this will set request to MPI_REQUEST_NULL if the message
      // is delivered, which causes the
      int err = libmpi.MPI_Test(request_, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        NVTX_MARK("Irecv:: MPI -> CUDA");
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
    while (state_ != State::CUDA) {
      wake();
    }
    CUDA_RUNTIME(cudaEventSynchronize(event_));
    return MPI_SUCCESS;
  }

  virtual bool needs_wake() override { return state_ != State::CUDA; }
};

namespace async {

void start_isend(Packer &packer, PARAMS_MPI_Isend) {
  std::unique_ptr<Isend> op = std::make_unique<Isend>(packer, ARGS_MPI_Isend);
  LOG_SPEW("managed Isend, req=" << uintptr_t(*request));
  active[*request] = std::move(op);
}

void start_irecv(Packer &packer, PARAMS_MPI_Irecv) {
  std::unique_ptr<Irecv> op = std::make_unique<Irecv>(packer, ARGS_MPI_Irecv);
  LOG_SPEW("managed Irecv, req=" << uintptr_t(*request));
  active[*request] = std::move(op);
}

int wait(MPI_Request *request, MPI_Status *status) {

  auto ii = active.find(*request);
  if (active.end() != ii) {
    LOG_SPEW("async::wait() on managed request " << uintptr_t(*request));
    int err = ii->second->wait(status);
    active.erase(ii);
    LOG_SPEW("async::wait() cleaned up request "
             << uintptr_t(request) << "(" << active.size() << " remaining)");
    return err;
  } else {
    LOG_SPEW("MPI_Wait on unmanaged request " << uintptr_t(*request));
    return libmpi.MPI_Wait(request, status);
  }
}

int try_progress() {
  NVTX_RANGE_PUSH("try progress");
  for (auto &kv : active) {
    if (kv.second->needs_wake()) {
      int err = kv.second->wake();
      if (err != MPI_SUCCESS) {
        return err;
      }
    }
  }
  NVTX_RANGE_POP();
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