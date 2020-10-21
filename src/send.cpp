#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "symbols.hpp"
#include "types.hpp"

#include "allocators.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <dlfcn.h>

#include <vector>

static int pack_gpu_gpu_unpack(int device, std::shared_ptr<Packer> packer,
                        PARAMS_MPI_Send) {
  CUDA_RUNTIME(cudaSetDevice(device));

  // reserve intermediate buffer
  int packedBytes;
  {
    int tySize;
    MPI_Type_size(datatype, &tySize);
    packedBytes = tySize * count;
  }
  void *packBuf = nullptr;
  packBuf = deviceAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B device send buffer");

  // pack into device buffer
  int pos = 0;
  packer->pack(packBuf, &pos, buf, count);

  // send to other device
  int err = libmpi.MPI_Send(packBuf, packedBytes, MPI_BYTE, dest, tag, comm);

  // release temporary buffer
  deviceAllocator.deallocate(packBuf, 0);

  return err;
}

extern "C" int MPI_Send(PARAMS_MPI_Send) {
  if (environment::noTempi) {
    libmpi.MPI_Send(ARGS_MPI_Send);
  }
  LOG_DEBUG("MPI_Send");

  // use library MPI for memory we can't reach on the device
  cudaPointerAttributes attr = {};
  CUDA_RUNTIME(cudaPointerGetAttributes(&attr, buf));
  if (nullptr == attr.devicePointer) {
    LOG_DEBUG("use library (host memory)");
    return libmpi.MPI_Send(ARGS_MPI_Send);
  }

  // optimize for fast GPU packer
  if (packerCache.count(datatype)) {
    std::shared_ptr<Packer> packer = packerCache[datatype];
    return pack_gpu_gpu_unpack(attr.device, packer, ARGS_MPI_Send);
  }

  // if all else fails, just do MPI_Send
  return libmpi.MPI_Send(ARGS_MPI_Send);
}
