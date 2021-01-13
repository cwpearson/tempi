//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "alltoallv_impl.hpp"

#include "allocators.hpp"
#include "cuda_runtime.hpp"
#include "logging.hpp"
#include "topology.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <vector>

/* implement as a bunch of Isend/Irecv, with the remote ones issued before the
 * local ones.
 */
int alltoallv_isir_remote_first(PARAMS_MPI_Alltoallv) {

  int err = MPI_SUCCESS;

  int commSize = 0;
  MPI_Comm_size(comm, &commSize);

  std::vector<MPI_Request> sendReqs(commSize, {});
  std::vector<MPI_Request> recvReqs(commSize, {});

  // start remote comms first. if this isn't MPI_COMM_WORLD, it will still be
  // correct, just random which are first and second
  for (int j = 0; j < commSize; ++j) {
    if (!is_colocated(comm, j)) {
      MPI_Isend(((char *)sendbuf) + sdispls[j], sendcounts[j], sendtype, j, 0,
                comm, &sendReqs[j]);
    }
  }
  for (int j = 0; j < commSize; ++j) {
    if (is_colocated(comm, j)) {
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
int alltoallv_staged(PARAMS_MPI_Alltoallv) {
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
int alltoallv_isir_staged(PARAMS_MPI_Alltoallv) {

  int err = MPI_SUCCESS;

  int commSize = 0;
  MPI_Comm_size(comm, &commSize);

  nvtxRangePush("alloc");
  size_t sendBufSize = sdispls[commSize - 1] + sendcounts[commSize - 1];
  size_t recvBufSize = rdispls[commSize - 1] + recvcounts[commSize - 1];
  LOG_SPEW("request " << sendBufSize << " for host sendbuf");
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
int alltoallv_isir_remote_staged(PARAMS_MPI_Alltoallv) {

  // TODO:
  // caching layer for pinned allocations
  // only copy remote data to host

  int err = MPI_SUCCESS;

  int commSize = 0;
  MPI_Comm_size(comm, &commSize);

  char *hSendBuf = {};
  char *hRecvBuf = {};
  size_t sendBufSize = 0, recvBufSize = 0;

  // precompute remote and local ranks
  std::vector<int> remotes, locals;
  remotes.reserve(commSize);
  locals.reserve(commSize);
  for (int j = 0; j < commSize; ++j) {
    if (!is_colocated(comm, j)) {
      remotes.push_back(j);
    } else {
      locals.push_back(j);
    }
  }

  nvtxRangePush("alloc");
  if (!remotes.empty()) {
    sendBufSize = sdispls[commSize - 1] + sendcounts[commSize - 1];
    recvBufSize = rdispls[commSize - 1] + recvcounts[commSize - 1];
    hSendBuf = hostAllocator.allocate(sendBufSize);
    hRecvBuf = hostAllocator.allocate(recvBufSize);
  }
  nvtxRangePop();

  // copy remote sends to host
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

  // copy remote recvs to device
  for (int j : remotes) {
    CUDA_RUNTIME(cudaMemcpy(((char *)recvbuf) + rdispls[j],
                            hRecvBuf + rdispls[j], recvcounts[j],
                            cudaMemcpyHostToDevice));
  }

  nvtxRangePush("free");
  hostAllocator.deallocate(hSendBuf, sendBufSize);
  hostAllocator.deallocate(hRecvBuf, recvBufSize);
  nvtxRangePop();

  return err;
}
