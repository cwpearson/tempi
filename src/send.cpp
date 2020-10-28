#include "allocators.hpp"
#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "symbols.hpp"
#include "topology.hpp"
#include "types.hpp"

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

static int staged(int numBytes, // pre-computed buffer size in bytes
                  PARAMS_MPI_Send) {

  // reserve intermediate buffer
  void *hostBuf = hostAllocator.allocate(numBytes);

  // copy to host
  CUDA_RUNTIME(cudaMemcpy(hostBuf, buf, numBytes, cudaMemcpyDeviceToHost));

  // send to other device
  int err = libmpi.MPI_Send(hostBuf, count, datatype, dest, tag, comm);

  // release temporary buffer
  hostAllocator.deallocate(hostBuf, numBytes);

  return err;
}

extern "C" int MPI_Send(PARAMS_MPI_Send) {
  if (environment::noTempi) {
    return libmpi.MPI_Send(ARGS_MPI_Send);
  }

  dest = topology::library_rank(comm, dest);

  // use library MPI for memory we can't reach on the device
  cudaPointerAttributes attr = {};
  CUDA_RUNTIME(cudaPointerGetAttributes(&attr, buf));
  if (nullptr == attr.devicePointer) {
    LOG_SPEW("MPI_Send: use library (host memory)");
    return libmpi.MPI_Send(ARGS_MPI_Send);
  }

  // optimize for fast GPU packer
  if (packerCache.count(datatype)) {
    LOG_SPEW("MPI_Send: pack_gpu_gpu_unpack");
    std::shared_ptr<Packer> packer = packerCache[datatype];
    return pack_gpu_gpu_unpack(attr.device, packer, ARGS_MPI_Send);
  }

  // message size
  int numBytes;
  {
    int tySize;
    MPI_Type_size(datatype, &tySize);
    numBytes = tySize * count;
  }

  // use staged for big remote messages
  if (!is_colocated(comm, dest) && numBytes >= (1 << 19) &&
      numBytes < (1 << 21)) {
    LOG_SPEW("MPI_Send: staged");
    return staged(numBytes, ARGS_MPI_Send);
  }

  // if all else fails, just do MPI_Send
  LOG_SPEW("MPI_Send: use library (fallthrough)");
  return libmpi.MPI_Send(ARGS_MPI_Send);
}
