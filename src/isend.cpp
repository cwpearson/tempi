#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "requests.hpp"
#include "symbols.hpp"
#include "types.hpp"
#include "worker.hpp"

#include "allocator_slab.hpp"

#include <cuda_runtime.h>
#include <mpi.h>
#include <nvToolsExt.h>

#include <dlfcn.h>

#include <vector>

extern "C" int MPI_Isend(PARAMS_MPI_Isend) {
  static Func_MPI_Isend fn = libmpi.MPI_Isend;
  if (environment::noTempi) {
    return fn(ARGS_MPI_Isend);
  }
  return fn(ARGS_MPI_Isend);

#if 0
  nvtxRangePush("MPI_Isend");
  LOG_DEBUG("MPI_Isend");

  // use library MPI for memory we can't reach on the device
  cudaPointerAttributes attr = {};
  CUDA_RUNTIME(cudaPointerGetAttributes(&attr, buf));
  if (nullptr == attr.devicePointer) {
    LOG_DEBUG("use library (host memory)");
    int err = fn(ARGS_MPI_Isend);
    nvtxRangePop();
    return err;
  }

  // use library MPI if we don't have a fast packer
  if (datatype != MPI_FLOAT && !packerCache.count(datatype)) {
    LOG_DEBUG("use library (no fast packer)");
    int err = fn(ARGS_MPI_Isend);
    nvtxRangePop();
    return err;
  }

  // we can use a background thread with not MPI_THREAD_SINGLE

  // FIXME:
  // we need to make sure library MPI_Isend is called before library MPI_Wait.
  // it may be a while before library MPI_Isend is called, even though we return
  // immediately and app may call MPI_Wait may be called right away. so, we need
  // to track that this request should not be passed onto the library MPI_Wait
  // yet. and MPI_Wait needs to make sure

  block_request(request);

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
  LOG_SPEW("blocking request " << uintptr_t(request));
  block_request(request);

  worker_push(job);


  nvtxRangePop();
  return MPI_SUCCESS;
#endif
}
