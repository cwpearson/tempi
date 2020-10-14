#include "cuda_runtime.hpp"
#include "logging.hpp"
#include "env.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <dlfcn.h>

#include <vector>

#define PARAMS                                                                 \
  void *buf, int count, MPI_Datatype datatype, int source, int tag,            \
      MPI_Comm comm, MPI_Status *status
#define ARGS buf, count, datatype, source, tag, comm, status

extern "C" int MPI_Recv(PARAMS) {

  // find the underlying MPI call
  typedef int (*Func_MPI_Recv)(PARAMS);
  static Func_MPI_Recv fn = nullptr;
  if (!fn) {
    fn = reinterpret_cast<Func_MPI_Recv>(dlsym(RTLD_NEXT, "MPI_Recv"));
  }
  TEMPI_DISABLE_GUARD;
  LOG_DEBUG("MPI_Recv");

  // use library MPI for memory we can't reach on the device
  cudaPointerAttributes attr = {};
  CUDA_RUNTIME(cudaPointerGetAttributes(&attr, buf));
  if (nullptr == attr.devicePointer) {
    LOG_DEBUG("use library (host memory)");
    return fn(ARGS);
  }

  return MPI_ERR_UNKNOWN;
}
