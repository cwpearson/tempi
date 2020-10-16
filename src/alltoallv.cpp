#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "topology.hpp"
#include "types.hpp"

#include "allocator_slab.hpp"

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvToolsExt.h>

#include <dlfcn.h>

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

  std::vector<MPI_Request> sendReqs(commSize, {});
  std::vector<MPI_Request> recvReqs(commSize, {});

  nvtxRangePush("alloc");
  size_t sendBufSize = sdispls[commSize - 1] + sendcounts[commSize - 1];
  size_t recvBufSize = rdispls[commSize - 1] + recvcounts[commSize - 1];
  char *hSendBuf = new char[sendBufSize];
  char *hRecvBuf = new char[recvBufSize];
  nvtxRangePop();

  CUDA_RUNTIME(
      cudaMemcpy(hSendBuf, sendbuf, sendBufSize, cudaMemcpyDeviceToHost));
  err = MPI_Alltoallv(hSendBuf, sendcounts, sdispls, sendtype, hRecvBuf, recvcounts,
                rdispls, recvtype, comm);
  CUDA_RUNTIME(
      cudaMemcpy(recvbuf, hRecvBuf, recvBufSize, cudaMemcpyHostToDevice));

  nvtxRangePush("free");
  delete[] hSendBuf;
  delete[] hRecvBuf;
  nvtxRangePop();

  return err;
}

extern "C" int MPI_Alltoallv(PARAMS) {
  typedef int (*Func_MPI_Alltoallv)(PARAMS);
  static Func_MPI_Alltoallv fn = nullptr;
  if (!fn) {
    fn =
        reinterpret_cast<Func_MPI_Alltoallv>(dlsym(RTLD_NEXT, "MPI_Alltoallv"));
  }
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
    return alltoallv_staged(ARGS);
  }
}
