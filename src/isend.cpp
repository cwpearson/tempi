#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "types.hpp"
#include "worker.hpp"

#include "allocator_slab.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <dlfcn.h>

#include <vector>

#define PARAMS                                                                 \
  const void *buf, int count, MPI_Datatype datatype, int dest, int tag,        \
      MPI_Comm comm, MPI_Request *request
#define ARGS buf, count, datatype, dest, tag, comm, request

extern "C" int MPI_Isend(PARAMS) {
  typedef int (*Func_MPI_Isend)(PARAMS);
  static Func_MPI_Isend fn = nullptr;
  if (!fn) {
    fn = reinterpret_cast<Func_MPI_Isend>(dlsym(RTLD_NEXT, "MPI_Isend"));
  }
  TEMPI_DISABLE_GUARD;
  LOG_DEBUG("MPI_Isend");

  // use library MPI for memory we can't reach on the device
  cudaPointerAttributes attr = {};
  CUDA_RUNTIME(cudaPointerGetAttributes(&attr, buf));
  if (nullptr == attr.devicePointer) {
    LOG_DEBUG("use library (host memory)");
    return fn(ARGS);
  }

  // use library MPI if we don't have a fast packer
  if (!packerCache.count(datatype)) {
    LOG_DEBUG("use library (no fast packer)");
    return fn(ARGS);
  }

  // FIXME:
  // we need to make sure library MPI_Isend is called before library MPI_Wait.
  // it may be a while before library MPI_Isend is called, even though we return
  // immediately and app may call MPI_Wait may be called right away. so, we need
  // to track that this request should not be passed onto the library MPI_Wait
  // yet. and MPI_Wait needs to make sure

  WorkerJob job;
  job.kind = WorkerJob::ISEND;
  WorkerJob::IsendParams params{.buf = buf,
                                .count = count,
                                .datatype = datatype,
                                .dest = dest,
                                .tag = tag,
                                .comm = comm,
                                .request = request};
  job.params.isend = params;

  return MPI_SUCCESS;
}
