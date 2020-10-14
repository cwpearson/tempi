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

  CUDA_RUNTIME(cudaSetDevice(attr.device));
  std::shared_ptr<Packer> packer = packerCache[datatype];

  // recv into device buffer
  int packedBytes;
  {
    int tySize;
    MPI_Type_size(datatype, &tySize);
    packedBytes = tySize * count;
  }
  void *packBuf = nullptr;
  // CUDA_RUNTIME(cudaMalloc(&packBuf, packedBytes));
  packBuf = testAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B device recv buffer");

  // send to other device
  int err = fn(packBuf, packedBytes, MPI_BYTE, source, tag, comm, status);

  // unpack from temporary buffer
  int pos = 0;
  packer->unpack(packBuf, &pos, buf, count);

  // release temporary buffer
  LOG_SPEW("free intermediate recv buffer");
  // CUDA_RUNTIME(cudaFree(packBuf));
  testAllocator.deallocate(packBuf, 0);

  return err;
}
