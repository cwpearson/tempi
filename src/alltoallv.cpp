#include "alltoallv_impl.hpp"
#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "symbols.hpp"

#include <cuda_runtime.h>

extern "C" int MPI_Alltoallv(PARAMS_MPI_Alltoallv) {
  static Func_MPI_Alltoallv fn = libmpi.MPI_Alltoallv;
  if (environment::noTempi) {
    return fn(ARGS_MPI_Alltoallv);
  }
  if (AlltoallvMethod::NONE == environment::alltoallv) {
    LOG_SPEW("MPI_Alltoallv: use library (TEMPI_ALLTOALLV_NONE)");
    return fn(ARGS_MPI_Alltoallv);
  }

  /* use library MPI for memory we can't reach on the device
    if a zero-size allocation is called with cudaMalloc, the null pointer will
    be returned, which we cannot detect as a device pointer.
    FIXME: for now, call the nullpointer a device pointer.
  */
  cudaPointerAttributes sendAttr = {}, recvAttr = {};
  CUDA_RUNTIME(cudaPointerGetAttributes(&sendAttr, sendbuf));
  CUDA_RUNTIME(cudaPointerGetAttributes(&recvAttr, recvbuf));
  if ((nullptr != sendbuf && nullptr == sendAttr.devicePointer) ||
      (nullptr != recvbuf && nullptr == recvAttr.devicePointer)) {
    LOG_SPEW("MPI_Alltoallv: use library (host memory)");
    return fn(ARGS_MPI_Alltoallv);
  }

  if (MPI_COMM_WORLD != comm) {
    LOG_SPEW("MPI_Alltoallv: use library (not MPI_COMM_WORLD)");
    return fn(ARGS_MPI_Alltoallv);
  }

  switch (environment::alltoallv) {
  case AlltoallvMethod::AUTO: {
    LOG_SPEW("MPI_Alltoallv: TEMPI_ALLTOALLV_AUTO");
    return alltoallv_staged(ARGS_MPI_Alltoallv);
  }
  case AlltoallvMethod::REMOTE_FIRST: {
    LOG_SPEW("MPI_Alltoallv: TEMPI_ALLTOALLV_REMOTE_FIRST");
    return alltoallv_isir_remote_first(ARGS_MPI_Alltoallv);
  }
  case AlltoallvMethod::STAGED: {
    LOG_SPEW("MPI_Alltoallv: TEMPI_ALLTOALLV_STAGED");
    return alltoallv_staged(ARGS_MPI_Alltoallv);
  }
  case AlltoallvMethod::ISIR_STAGED: {
    LOG_SPEW("MPI_Alltoallv: TEMPI_ALLTOALLV_ISIR_STAGED");
    return alltoallv_isir_staged(ARGS_MPI_Alltoallv);
  }
  case AlltoallvMethod::ISIR_REMOTE_STAGED: {
    LOG_SPEW("MPI_Alltoallv: TEMPI_ALLTOALLV_ISIR_REMOTE_STAGED");
    return alltoallv_isir_remote_staged(ARGS_MPI_Alltoallv);
  }
  case AlltoallvMethod::NONE:
  default:
    LOG_FATAL("unexpected AlltoallvMethod");
  }
}
