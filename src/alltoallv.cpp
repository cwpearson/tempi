#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "symbols.hpp"
#include "topology.hpp"
#include "types.hpp"

#include "allocators.hpp"

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvToolsExt.h>

#include <dlfcn.h>

#include <chrono>
#include <thread>
#include <vector>

#define PARAMS                                                                 \
  const void *sendbuf, const int sendcounts[], const int sdispls[],            \
      MPI_Datatype sendtype, void *recvbuf, const int recvcounts[],            \
      const int rdispls[], MPI_Datatype recvtype, MPI_Comm comm
#define ARGS                                                                   \
  sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,        \
      recvtype, comm

/* implement as a bunch of Isend/Irecv, with the remote ones issued before the
 * local ones
 */
int alltoallv_remote_first(PARAMS) {

  int err = MPI_SUCCESS;

  int commSize = 0;
  MPI_Comm_size(comm, &commSize);

  std::vector<MPI_Request> sendReqs(commSize, {});
  std::vector<MPI_Request> recvReqs(commSize, {});

  // start remote comms first. if this isn't MPI_COMM_WORLD, it will still be
  // correct, just random which are first and second
  for (int j = 0; j < commSize; ++j) {
    if (!is_colocated(j)) {
      MPI_Isend(((char *)sendbuf) + sdispls[j], sendcounts[j], sendtype, j, 0,
                comm, &sendReqs[j]);
    }
  }
  for (int j = 0; j < commSize; ++j) {
    if (is_colocated(j)) {
      MPI_Isend(((char *)sendbuf) + sdispls[j], sendcounts[j], sendtype, j, 0,
                comm, &sendReqs[j]);
    }
  }

  for (int i = 0; i < commSize; ++i) {
    {
      int e = MPI_Irecv(((char *)recvbuf) + rdispls[i], recvcounts[i], recvtype,
                        i, 0, comm, &recvReqs[i]);
      err = (MPI_SUCCESS == e ? err : e);
    }
  }
  {
    int e = MPI_Waitall(sendReqs.size(), sendReqs.data(), MPI_STATUS_IGNORE);
    err = (MPI_SUCCESS == e ? err : e);
  }
  {
    int e = MPI_Waitall(recvReqs.size(), recvReqs.data(), MPI_STATUS_IGNORE);
    err = (MPI_SUCCESS == e ? err : e);
  }

  return err;
}

/* implement as a bunch of copies to the CPU, then a CPU alltoallv, then a bunch
 * of copies to the GPU
 */
int alltoallv_staged(PARAMS) {
  int err = MPI_SUCCESS;

  int commSize = 0;
  MPI_Comm_size(comm, &commSize);

  nvtxRangePush("alloc");
  size_t sendBufSize = sdispls[commSize - 1] + sendcounts[commSize - 1];
  size_t recvBufSize = rdispls[commSize - 1] + recvcounts[commSize - 1];
  char *hSendBuf = hostAllocator.allocate(sendBufSize);
  char *hRecvBuf = hostAllocator.allocate(recvBufSize);
  nvtxRangePop();

  CUDA_RUNTIME(
      cudaMemcpy(hSendBuf, sendbuf, sendBufSize, cudaMemcpyDeviceToHost));
  err = libmpi.MPI_Alltoallv(hSendBuf, sendcounts, sdispls, sendtype, hRecvBuf,
                             recvcounts, rdispls, recvtype, comm);
  CUDA_RUNTIME(
      cudaMemcpy(recvbuf, hRecvBuf, recvBufSize, cudaMemcpyHostToDevice));

  nvtxRangePush("free");
  hostAllocator.deallocate(hSendBuf, sendBufSize);
  hostAllocator.deallocate(hRecvBuf, recvBufSize);
  nvtxRangePop();
  return err;
}

/* copy to host, and do a bunch of isend/irecv
 */
int alltoallv_staged_isir(PARAMS) {

  int err = MPI_SUCCESS;

  int commSize = 0;
  MPI_Comm_size(comm, &commSize);

  nvtxRangePush("alloc");
  size_t sendBufSize = sdispls[commSize - 1] + sendcounts[commSize - 1];
  size_t recvBufSize = rdispls[commSize - 1] + recvcounts[commSize - 1];
  char *hSendBuf = hostAllocator.allocate(sendBufSize);
  char *hRecvBuf = hostAllocator.allocate(recvBufSize);
  nvtxRangePop();

  std::vector<MPI_Request> sendReqs(commSize, {});
  std::vector<MPI_Request> recvReqs(commSize, {});

  CUDA_RUNTIME(
      cudaMemcpy(hSendBuf, sendbuf, sendBufSize, cudaMemcpyDeviceToHost));

  // start remote comms first. if this isn't MPI_COMM_WORLD, it will still be
  // correct, just random which are first and second
  for (int j = 0; j < commSize; ++j) {
    MPI_Isend(((char *)hSendBuf) + sdispls[j], sendcounts[j], sendtype, j, 0,
              comm, &sendReqs[j]);
  }
  for (int i = 0; i < commSize; ++i) {
    {
      int e = MPI_Irecv(((char *)hRecvBuf) + rdispls[i], recvcounts[i],
                        recvtype, i, 0, comm, &recvReqs[i]);
      err = (MPI_SUCCESS == e ? err : e);
    }
  }
  {
    int e = MPI_Waitall(sendReqs.size(), sendReqs.data(), MPI_STATUS_IGNORE);
    err = (MPI_SUCCESS == e ? err : e);
  }
  {
    int e = MPI_Waitall(recvReqs.size(), recvReqs.data(), MPI_STATUS_IGNORE);
    err = (MPI_SUCCESS == e ? err : e);
  }

  CUDA_RUNTIME(
      cudaMemcpy(recvbuf, hRecvBuf, recvBufSize, cudaMemcpyHostToDevice));

  nvtxRangePush("free");
  hostAllocator.deallocate(hSendBuf, sendBufSize);
  hostAllocator.deallocate(hRecvBuf, recvBufSize);
  nvtxRangePop();

  return err;
}

/* implement as isends/irecvs.
   local is cuda-aware, remote is staged through host
 */
int alltoallv_isend_irecv(PARAMS) {

  // TODO:
  // caching layer for pinned allocations
  // only copy remote data to host

  int err = MPI_SUCCESS;

  int commSize = 0;
  MPI_Comm_size(comm, &commSize);

  char *hSendBuf = {};
  char *hRecvBuf = {};

  // precompute remote and local ranks
  std::vector<int> remotes, locals;
  remotes.reserve(commSize);
  locals.reserve(commSize);
  for (int j = 0; j < commSize; ++j) {
    if (!is_colocated(j)) {
      remotes.push_back(j);
    } else {
      locals.push_back(j);
    }
  }

  nvtxRangePush("alloc");
  if (!remotes.empty()) {
    size_t sendBufSize = sdispls[commSize - 1] + sendcounts[commSize - 1];
    size_t recvBufSize = rdispls[commSize - 1] + recvcounts[commSize - 1];
    CUDA_RUNTIME(cudaMallocHost(&hSendBuf, sendBufSize));
    CUDA_RUNTIME(cudaMallocHost(&hRecvBuf, recvBufSize));
  }
  nvtxRangePop();

  // copy remote messages to host
  for (int j : remotes) {
    CUDA_RUNTIME(cudaMemcpy(hSendBuf + sdispls[j],
                            ((char *)sendbuf) + sdispls[j], sendcounts[j],
                            cudaMemcpyDeviceToHost));
  }

  std::vector<MPI_Request> sendReqs(commSize, {});
  std::vector<MPI_Request> recvReqs(commSize, {});

  // send remote from the host
  for (int j : remotes) {
    nvtxRangePush("MPI_Isend (remote)");
    MPI_Isend(((char *)hSendBuf) + sdispls[j], sendcounts[j], sendtype, j, 0,
              comm, &sendReqs[j]);
    nvtxRangePop();
  }
  // send local direct from the device
  for (int j : locals) {
    nvtxRangePush("MPI_Isend (local)");
    MPI_Isend(((char *)sendbuf) + sdispls[j], sendcounts[j], sendtype, j, 0,
              comm, &sendReqs[j]);
    nvtxRangePop();
  }

  // recv local to the device
  for (int i : locals) {
    nvtxRangePush("MPI_Irecv (local)");
    int e = MPI_Irecv(((char *)recvbuf) + rdispls[i], recvcounts[i], recvtype,
                      i, 0, comm, &recvReqs[i]);
    nvtxRangePop();
    err = (MPI_SUCCESS == e ? err : e);
  }
  // recv remote to the host
  for (int i : remotes) {
    nvtxRangePush("MPI_Irecv (remote)");
    int e = MPI_Irecv(((char *)hRecvBuf) + rdispls[i], recvcounts[i], recvtype,
                      i, 0, comm, &recvReqs[i]);
    nvtxRangePop();
    err = (MPI_SUCCESS == e ? err : e);
  }

  {
    nvtxRangePush("waitall (send)");
    int e = MPI_Waitall(sendReqs.size(), sendReqs.data(), MPI_STATUS_IGNORE);
    nvtxRangePop();
    err = (MPI_SUCCESS == e ? err : e);
  }
  {
    nvtxRangePush("waitall (recv)");
    int e = MPI_Waitall(recvReqs.size(), recvReqs.data(), MPI_STATUS_IGNORE);
    nvtxRangePop();
    err = (MPI_SUCCESS == e ? err : e);
  }

  for (int j : remotes) {
    CUDA_RUNTIME(cudaMemcpy(((char *)recvbuf) + rdispls[j],
                            hRecvBuf + rdispls[j], recvcounts[j],
                            cudaMemcpyHostToDevice));
  }

  nvtxRangePush("free");
  CUDA_RUNTIME(cudaFreeHost(hSendBuf));
  CUDA_RUNTIME(cudaFreeHost(hRecvBuf));
  nvtxRangePop();

  return err;
}

extern "C" int MPI_Alltoallv(PARAMS) {
  static Func_MPI_Alltoallv fn = libmpi.MPI_Alltoallv;
  TEMPI_DISABLE_GUARD;
  if (environment::noAlltoallv) {
    return fn(ARGS);
  }
  LOG_DEBUG("MPI_Alltoallv");

  // use library MPI for memory we can't reach on the device
  cudaPointerAttributes sendAttr = {}, recvAttr = {};
  CUDA_RUNTIME(cudaPointerGetAttributes(&sendAttr, sendbuf));
  CUDA_RUNTIME(cudaPointerGetAttributes(&recvAttr, recvbuf));
  if (nullptr == sendAttr.devicePointer || nullptr == recvAttr.devicePointer) {
    LOG_DEBUG("use library (host memory)");
    return fn(ARGS);
  }

  if (MPI_COMM_WORLD != comm) {
    LOG_WARN("alltoallv remote-first optimization disabled");
    return fn(ARGS);
  } else {
    // return alltoallv_remote_first(ARGS);
    // return alltoallv_staged(ARGS);
    return alltoallv_staged_isir(ARGS);
    // return alltoallv_isend_irecv(ARGS);
  }
}
