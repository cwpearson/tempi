#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "types.hpp"

#include "allocator_slab.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <dlfcn.h>

#include <vector>

#define PARAMS                                                                 \
  const void *buf, int count, MPI_Datatype datatype, int dest, int tag,        \
      MPI_Comm comm
#define ARGS buf, count, datatype, dest, tag, comm

extern "C" int MPI_Send(PARAMS) {
  typedef int (*Func_MPI_Send)(PARAMS);
  static Func_MPI_Send fn = nullptr;
  if (!fn) {
    fn = reinterpret_cast<Func_MPI_Send>(dlsym(RTLD_NEXT, "MPI_Send"));
  }
  TEMPI_DISABLE_GUARD;
  LOG_DEBUG("MPI_Send");

  // use library MPI if we don't have a fast packer
  if (!packerCache.count(datatype)) {
    LOG_DEBUG("use library (no fast packer)");
    return fn(ARGS);
  }

  // use library MPI for memory we can't reach on the device
  cudaPointerAttributes attr = {};
  CUDA_RUNTIME(cudaPointerGetAttributes(&attr, buf));
  if (nullptr == attr.devicePointer) {
    LOG_DEBUG("use library (host memory)");
    return fn(ARGS);
  }

  CUDA_RUNTIME(cudaSetDevice(attr.device));
  std::shared_ptr<Packer> packer = packerCache[datatype];

  // reserve intermediate buffer
  int packedBytes;
  {
    int tySize;
    MPI_Type_size(datatype, &tySize);
    packedBytes = tySize * count;
  }
  void *packBuf = nullptr;
  // CUDA_RUNTIME(cudaMalloc(&packBuf, packedBytes));
  packBuf = testAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B device send buffer");

  // pack into device buffer
  int pos = 0;
  packer->pack(packBuf, &pos, buf, count);

  // send to other device
  int err = fn(packBuf, packedBytes, MPI_BYTE, dest, tag, comm);

  // release temporary buffer
  // CUDA_RUNTIME(cudaFree(packBuf));
  testAllocator.deallocate(packBuf, 0);

  return err;
}
