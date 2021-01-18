//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "async_operation.hpp"

#include "allocators.hpp"
#include "counters.hpp"
#include "cuda_runtime.hpp"
#include "events.hpp"
#include "logging.hpp"
#include "request.hpp"
#include "symbols.hpp"

#include <map>
#include <memory>

//#define USE_NVTX
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

Isend/Irecv manage the MPI request internally, since the lifetime of the TEMPI
Isend/Irecv is longer than the lifetime of the MPI Isend/Irecv

TEMPI provides a fake request to the caller as a handle.
*/
class Isend : public AsyncOperation {
  Packer &packer_;
  MPI_Request
      request_; // copy of the provided request - see class documentation

  enum class State { CUDA, MPI };
  State state_;

  void *packedBuf_;
  int packedSize_;

  cudaEvent_t event_;

public:
  Isend(Packer &packer, PARAMS_MPI_Isend)
      : packer_(packer), state_(State::CUDA), packedBuf_(nullptr),
        packedSize_(0) {
    NVTX_MARK("Isend()");
    event_ = events::request();

    // allocate intermediate space
    MPI_Pack_size(count, datatype, comm, &packedSize_);
    packedBuf_ = hostAllocator.allocate(packedSize_);
    // LOG_SPEW("buffer is @" << uintptr_t(packedBuf_));

    // issue pack operation
    int position = 0;
    packer.pack_async(packedBuf_, &position, buf, count, event_);
    LOG_SPEW("Isend():: issued pack");

    // initialize Isend with internal request
    {
      TEMPI_COUNTER_OP(libCalls, SEND_INIT_NUM, ++);
      double start = MPI_Wtime();
      libmpi.MPI_Send_init(packedBuf_, packedSize_, MPI_PACKED, dest, tag, comm,
                           &request_);
      TEMPI_COUNTER_OP(libCalls, SEND_INIT_TIME, += MPI_Wtime() - start);
    }

    // provide caller with a new TEMPI request
    *request = Request::make();

    LOG_SPEW("Isend():: init'ed Send, caller req=" << intptr_t(*request));
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
        LOG_SPEW(
            "Isend::wake() MPI_Start, internal req=" << intptr_t(request_));
        state_ = State::MPI;
        // manipulate local request, not Caller's copy
        {
          TEMPI_COUNTER_OP(libCalls, START_NUM, ++);
#ifdef TEMPI_ENABLE_COUNTERS
          double start = MPI_Wtime();
#endif
          const int merr = libmpi.MPI_Start(&request_);
          TEMPI_COUNTER_OP(libCalls, START_TIME, += MPI_Wtime() - start);
          return merr;
        }
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
    return libmpi.MPI_Wait(&request_, status);
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
  MPI_Request request_;
  void *buf_;
  int count_;

  enum class State { CUDA, MPI };
  State state_;

  void *packedBuf_;
  int packedSize_;

  cudaEvent_t event_;

public:
  Irecv(Packer &packer, PARAMS_MPI_Irecv)
      : packer_(packer), buf_(buf), count_(count), state_(State::MPI),
        packedBuf_(nullptr), packedSize_(0) {
    NVTX_MARK("Irecv()");
    event_ = events::request();

    // allocate intermediate space
    MPI_Pack_size(count, datatype, comm, &packedSize_);
    packedBuf_ = hostAllocator.allocate(packedSize_);

    // issue MPI_Irecv with internal request
    NVTX_RANGE_PUSH("MPI_Irecv");
    {
      TEMPI_COUNTER_OP(libCalls, IRECV_NUM, ++);
#ifdef TEMPI_ENABLE_COUNTERS
      double start = MPI_Wtime();
#endif
      libmpi.MPI_Irecv(packedBuf_, packedSize_, MPI_PACKED, source, tag, comm,
                       &request_);
      TEMPI_COUNTER_OP(libCalls, IRECV_TIME, += MPI_Wtime() - start);
    }
    NVTX_RANGE_POP();

    // give caller a new TEMPI request
    *request = Request::make();
    LOG_SPEW("Irecv(): issued Irecv, MPI req=" << intptr_t(request_));
  }
  ~Irecv() {
    hostAllocator.deallocate(packedBuf_, packedSize_);
    events::release(event_);
  }

  virtual int wake() override {
    NVTX_MARK("Irecv::wake()");
    switch (state_) {
    case State::MPI: {
      // TODO: handle status
      int flag;
      // if the MPI is complete this will set internal request to
      // MPI_REQUEST_NULL
      int err = libmpi.MPI_Test(&request_, &flag, MPI_STATUS_IGNORE);
      if (flag) {
        NVTX_MARK("Irecv:: MPI -> CUDA");
        // issue unpack operation
        int position = 0;
        packer_.unpack_async(packedBuf_, &position, buf_, count_, event_);
        state_ = State::CUDA;
      } else {
        NVTX_MARK("Irecv:: MPI not done");
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
  LOG_SPEW("managed Isend, caller req=" << intptr_t(*request));
  active[*request] = std::move(op);
}

void start_irecv(Packer &packer, PARAMS_MPI_Irecv) {
  std::unique_ptr<Irecv> op = std::make_unique<Irecv>(packer, ARGS_MPI_Irecv);
  LOG_SPEW("managed Irecv, caller req=" << intptr_t(*request));
  active[*request] = std::move(op);
}

int wait(MPI_Request *request, MPI_Status *status) {
  auto ii = active.find(*request);
  if (active.end() != ii) {
    LOG_SPEW("async::wait() on managed (caller) request "
             << intptr_t(*request));
    int err = ii->second->wait(status);
    active.erase(ii);
    LOG_SPEW("async::wait() cleaned up request "
             << intptr_t(*request) << "(" << active.size() << " remaining)");
    *request = MPI_REQUEST_NULL; // clear the caller's request
    return err;
  } else {
    LOG_SPEW("MPI_Wait on unmanaged request " << intptr_t(*request));
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
