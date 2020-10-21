#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "symbols.hpp"
#include "types.hpp"

#include "allocators.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <vector>

static int pack_gpu_gpu_unpack(int device, std::shared_ptr<Packer> packer,
                               PARAMS_MPI_Recv) {
  CUDA_RUNTIME(cudaSetDevice(device));

  // recv into device buffer
  int packedBytes;
  {
    int tySize;
    MPI_Type_size(datatype, &tySize);
    packedBytes = tySize * count;
  }
  void *packBuf = nullptr;
  packBuf = deviceAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B device recv buffer");

  // send to other device
  int err = libmpi.MPI_Recv(ARGS_MPI_Recv);

  // unpack from temporary buffer
  int pos = 0;
  packer->unpack(packBuf, &pos, buf, count);

  // release temporary buffer
  LOG_SPEW("free intermediate recv buffer");
  deviceAllocator.deallocate(packBuf, 0);
  return err;
}

extern "C" int MPI_Recv(PARAMS_MPI_Recv) {

  if (environment::noTempi) {
    libmpi.MPI_Recv(ARGS_MPI_Recv);
  }

  LOG_DEBUG("MPI_Recv");

  // use library MPI for memory we can't reach on the device
  cudaPointerAttributes attr = {};
  CUDA_RUNTIME(cudaPointerGetAttributes(&attr, buf));
  if (nullptr == attr.devicePointer) {
    LOG_DEBUG("use library (host memory)");
    return libmpi.MPI_Recv(ARGS_MPI_Recv);
  }

  // optimize packer
  if (packerCache.count(datatype)) {
    LOG_DEBUG("MPI_Recv: fast packer");
    std::shared_ptr<Packer> packer = packerCache[datatype];
    return pack_gpu_gpu_unpack(attr.device, packer, ARGS_MPI_Recv);
  }

  // if all else fails, just call MPI_Recv
  return libmpi.MPI_Recv(ARGS_MPI_Recv);
}
