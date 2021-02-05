//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "async_operation.hpp"

#include "allocators.hpp"
#include "counters.hpp"
#include "cuda_runtime.hpp"
#include "env.hpp"
#include "events.hpp"
#include "logging.hpp"
#include "measure_system.hpp"
#include "request.hpp"
#include "symbols.hpp"
#include "topology.hpp"

#include <unordered_map>
#include <memory>
#include <numeric> // iota

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
std::unordered_map<MPI_Request, std::unique_ptr<AsyncOperation>> active;

// how to implement the send
SystemPerformance::SendNonContigNd::MethodCache methodCache;

/*
Manages the state of a particular Isend
CUDA - the CUDA transfer is active or completion is not detected yet
MPI - the MPI_Isend has been issued

Isend/Irecv manage the MPI request internally, since the lifetime of the
TEMPI Isend/Irecv is longer than the lifetime of the MPI Isend/Irecv

TEMPI provides a fake request to the caller as a handle.

Instead of override this and putting the method specific stuff in calls,
everything is predicated in the ctor.
Can't call virtual functions in ctor, so we'd have to store the MPI_Isend params
in the class to propagate them through
TODO: this is probably worth doing ultimately.
*/
class Isend : public AsyncOperation {
public:
  using Method = SystemPerformance::SendNonContigNd::Method;
  using Args = SystemPerformance::SendNonContigNd::Args;

private:
  Packer &packer_;
  MPI_Request
      request_; // copy of the provided request - see class documentation

  enum class State { CUDA, MPI };
  State state_;

  void *packedBuf_;
  Method method_;
  int packSize_;

  cudaEvent_t event_;

public:
  Isend(Packer &packer, Method method,
        int packSize, // MPI_Pack_size
        PARAMS_MPI_Isend)
      : packer_(packer), state_(State::CUDA), packedBuf_(nullptr),
        method_(method), packSize_(packSize) {
    (void)datatype;
    NVTX_MARK("Isend()");
    event_ = events::request();

    // allocate intermediate space
    switch (method_) {
    case Method::ONESHOT: {
      TEMPI_COUNTER_OP(isend, NUM_ONESHOT, ++);
      packedBuf_ = hostAllocator.allocate(packSize_);
      break;
    }
    case Method::DEVICE: {
      TEMPI_COUNTER_OP(isend, NUM_DEVICE, ++);
      packedBuf_ = deviceAllocator.allocate(packSize_);
      break;
    }
    case Method::UNKNOWN:
    default:
      LOG_FATAL("unexpected method in isend");
    }

    // issue pack operation
    int position = 0;
    packer.pack_async(packedBuf_, &position, buf, count, event_);

    LOG_SPEW("Isend():: issued pack");

    // initialize Isend with internal request
    {
      TEMPI_COUNTER_OP(libCalls, SEND_INIT_NUM, ++);
      TEMPI_COUNTER_EXPR(double start = MPI_Wtime());
      libmpi.MPI_Send_init(packedBuf_, packSize_, MPI_PACKED, dest, tag, comm,
                           &request_);
      TEMPI_COUNTER_OP(libCalls, SEND_INIT_TIME, += MPI_Wtime() - start);
    }

    // provide caller with a new TEMPI request
    *request = Request::make();

    LOG_SPEW("Isend():: init'ed Send, caller req=" << intptr_t(*request));
  }
  ~Isend() {
    switch (method_) {
    case Method::ONESHOT: {
      hostAllocator.deallocate(packedBuf_, packSize_);
      break;
    }
    case Method::DEVICE: {
      deviceAllocator.deallocate(packedBuf_, packSize_);
      break;
    }
    case Method::UNKNOWN:
    default:
      LOG_FATAL("unexpected method in isend");
    }
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
public:
  using Method = SystemPerformance::SendNonContigNd::Method;
  using Args = SystemPerformance::SendNonContigNd::Args;

private:
  Packer &packer_;
  MPI_Request request_;
  void *buf_;
  int count_;

  enum class State { CUDA, MPI };
  State state_;

  void *packedBuf_;
  Method method_;
  int packSize_;

  cudaEvent_t event_;

public:
  Irecv(Packer &packer, Method method, int packSize, PARAMS_MPI_Irecv)
      : packer_(packer), buf_(buf), count_(count), state_(State::MPI),
        packedBuf_(nullptr), method_(method), packSize_(packSize) {
    (void)datatype;
    NVTX_MARK("Irecv()");
    event_ = events::request();

    // allocate intermediate space
    switch (method_) {
    case Method::ONESHOT: {
      TEMPI_COUNTER_OP(irecv, NUM_ONESHOT, ++);
      packedBuf_ = hostAllocator.allocate(packSize_);
      break;
    }
    case Method::DEVICE: {
      TEMPI_COUNTER_OP(irecv, NUM_DEVICE, ++);
      packedBuf_ = deviceAllocator.allocate(packSize_);
      break;
    }
    case Method::UNKNOWN:
    default:
      LOG_FATAL("unexpected method in isend");
    }

    // issue MPI_Irecv with internal request
    NVTX_RANGE_PUSH("MPI_Irecv");
    {
      TEMPI_COUNTER_OP(libCalls, IRECV_NUM, ++);
      TEMPI_COUNTER_EXPR(double start = MPI_Wtime());
      libmpi.MPI_Irecv(packedBuf_, packSize_, MPI_PACKED, source, tag, comm,
                       &request_);
      TEMPI_COUNTER_OP(libCalls, IRECV_TIME, += MPI_Wtime() - start);
    }
    NVTX_RANGE_POP();

    // give caller a new TEMPI request
    *request = Request::make();
    LOG_SPEW("Irecv(): issued Irecv, MPI req=" << intptr_t(request_));
  }
  ~Irecv() {
    switch (method_) {
    case Method::ONESHOT: {
      hostAllocator.deallocate(packedBuf_, packSize_);
      break;
    }
    case Method::DEVICE: {
      deviceAllocator.deallocate(packedBuf_, packSize_);
      break;
    }
    case Method::UNKNOWN:
    default:
      LOG_FATAL("unexpected method in isend");
    }
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

void start_isend(const StridedBlock &sb, Packer &packer, PARAMS_MPI_Isend) {

  // don't count this in modeling as we need it for intermediate alloc
  int packSize;
  MPI_Pack_size(count, datatype, comm, &packSize);

  Isend::Method method = Isend::Method::UNKNOWN;
  switch (environment::datatype) {
  case DatatypeMethod::AUTO: {
    // decide on one-shot or staged packing
    TEMPI_COUNTER_EXPR(double start = MPI_Wtime());
    const int blockLength = std::min(std::max(1, int(sb.counts[0])), 512);
    const bool colocated = is_colocated(comm, dest);
    Isend::Args args{
        .colocated = colocated, .bytes = packSize, .blockLength = blockLength};
    auto it = methodCache.find(args);
    if (methodCache.end() == it) {
      TEMPI_COUNTER_OP(modeling, CACHE_MISS, ++);
      nonstd::optional<double> d = systemPerformance.model_device(args);
      nonstd::optional<double> o = systemPerformance.model_oneshot(args);
      if (!o || !d) {
        method = Isend::Method::UNKNOWN;
      } else if (*o < *d) {
        method = Isend::Method::ONESHOT;
      } else {
        method = Isend::Method::DEVICE;
      }
      methodCache[args] = method;
    } else {
      TEMPI_COUNTER_OP(modeling, CACHE_HIT, ++);
      method = it->second;
    }
    TEMPI_COUNTER_OP(modeling, WALL_TIME, += MPI_Wtime() - start);
    break;
  }
  case DatatypeMethod::ONESHOT: {
    method = Isend::Method::ONESHOT;
    break;
  }
  case DatatypeMethod::DEVICE: {
    method = Isend::Method::DEVICE;
    break;
  }
  case DatatypeMethod::STAGED: {
    LOG_FATAL("TEMPI_DATATYPE_METHOD=STAGED not supported for Isend");
  }
  default: {
    LOG_FATAL("unexpected TEMPI_DATATYPE_METHOD");
  }
  }

  std::unique_ptr<Isend> op =
      std::make_unique<Isend>(packer, method, packSize, ARGS_MPI_Isend);
  LOG_SPEW("managed Isend, caller req=" << intptr_t(*request));
  active[*request] = std::move(op);
}

void start_irecv(const StridedBlock &sb, Packer &packer, PARAMS_MPI_Irecv) {

  // don't count this in modeling as we need it for intermediate alloc
  int packSize;
  MPI_Pack_size(count, datatype, comm, &packSize);

  Isend::Method method = Isend::Method::UNKNOWN;
  switch (environment::datatype) {
  case DatatypeMethod::AUTO: {
    // decide on one-shot or staged packing
    TEMPI_COUNTER_EXPR(double start = MPI_Wtime());
    const int blockLength = std::min(std::max(1, int(sb.counts[0])), 512);
    const bool colocated = is_colocated(comm, source);
    Isend::Args args{
        .colocated = colocated, .bytes = packSize, .blockLength = blockLength};
    auto it = methodCache.find(args);
    if (methodCache.end() == it) {
      TEMPI_COUNTER_OP(modeling, CACHE_MISS, ++);
      nonstd::optional<double> d = systemPerformance.model_device(args);
      nonstd::optional<double> o = systemPerformance.model_oneshot(args);
      if (!o || !d) {
        method = Isend::Method::UNKNOWN;
      } else if (*o < *d) {
        method = Isend::Method::ONESHOT;
      } else {
        method = Isend::Method::DEVICE;
      }
      methodCache[args] = method;
    } else {
      TEMPI_COUNTER_OP(modeling, CACHE_HIT, ++);
      method = it->second;
    }
    TEMPI_COUNTER_OP(modeling, WALL_TIME, += MPI_Wtime() - start);
    break;
  }
  case DatatypeMethod::ONESHOT: {
    method = Isend::Method::ONESHOT;
    break;
  }
  case DatatypeMethod::DEVICE: {
    method = Isend::Method::DEVICE;
    break;
  }
  case DatatypeMethod::STAGED: {
    LOG_FATAL("TEMPI_DATATYPE_METHOD=STAGED not supported for Irecv");
  }
  default: {
    LOG_FATAL("unexpected TEMPI_DATATYPE_METHOD");
  }
  }

  std::unique_ptr<Irecv> op =
      std::make_unique<Irecv>(packer, method, packSize, ARGS_MPI_Irecv);
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

#if 0
int waitall(PARAMS_MPI_Waitall) {

  std::vector<size_t> remaining(count);
  std::iota(remaining.begin(), remaining.end(), 0);

loop:
  while (!remaining.empty()) {

  wakeup_all:
    // give all operations an opportunity to progress
    // this means some can work while we wait on others
    for (size_t i : remaining) {
      auto it = active.find(array_of_requests[i]);
      if (active.end() != it && it->second->needs_wake()) {
        it->second->wake();
      }
    }

    // try to wait on an operation that does not need to wake up
    // if we succeed, give all operations a chance to wake up again
    for (size_t ri = 0; ri < remaining.size(); ++ri) {
      auto ai = active.find(array_of_requests[remaining[ri]]);
      if (active.end() != ai && !ai->second->needs_wake()) {
        ai->second->wait(&array_of_statuses[ri]);
        active.erase(ai);
        remaining.erase(remaining.begin() + ri);
        goto loop;
      }
    }

    // if we get here, no requests are managed by TEMPI
  }
}
#endif

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
