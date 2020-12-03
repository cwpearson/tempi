#include "allocators.hpp"
#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "packer_cache.hpp"
#include "symbols.hpp"
#include "topology.hpp"
#include "types.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <dlfcn.h>

#include <vector>

/* pack data into GPU buffer and send */
static int send_device(int device, Packer &packer, PARAMS_MPI_Send) {
  CUDA_RUNTIME(cudaSetDevice(device));

  // reserve intermediate buffer
  int packedBytes;
  {
    int size;
    MPI_Pack_size(count, datatype, comm, &size);
    packedBytes = size;
  }
  void *packBuf = nullptr;
  packBuf = deviceAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B device send buffer");

  // pack into device buffer
  int pos = 0;
  packer.pack(packBuf, &pos, buf, count);

  // send to other device
  int err = libmpi.MPI_Send(packBuf, packedBytes, MPI_PACKED, dest, tag, comm);

  // release temporary buffer
  deviceAllocator.deallocate(packBuf, packedBytes);

  return err;
}

/* pack data into pinned buffer and send */
static int send_oneshot(int device, Packer &packer, PARAMS_MPI_Send) {
  CUDA_RUNTIME(cudaSetDevice(device));

  // reserve intermediate buffer
  int packedBytes;
  {
    int size;
    MPI_Pack_size(count, datatype, comm, &size);
    packedBytes = size;
  }
  void *packBuf = nullptr;
  packBuf = hostAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B device send buffer");

  // pack into device buffer
  int pos = 0;
  packer.pack(packBuf, &pos, buf, count);

  // send to other device
  int err = libmpi.MPI_Send(packBuf, packedBytes, MPI_PACKED, dest, tag, comm);

  // release temporary buffer
  hostAllocator.deallocate(packBuf, packedBytes);

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
  auto pi = packerCache.find(datatype);
  if (packerCache.end() != pi) {

    switch (environment::datatype){
      case DatatypeMethod::ONESHOT: {
    LOG_SPEW("MPI_Send: send_oneshot");
    return send_oneshot(attr.device, *(pi->second), ARGS_MPI_Send);
      }
      case DatatypeMethod::DEVICE: {
    LOG_SPEW("MPI_Send: send_device");
    return send_device(attr.device, *(pi->second), ARGS_MPI_Send);
      }
      default:
      LOG_ERROR("unexpected DatatypeMethod");
      return MPI_ERR_UNKNOWN;
    }
  } else {
    LOG_SPEW("MPI_Send: no packer for " << uintptr_t(datatype));
  }

  // message size
  int numBytes;
  {
    int tySize;
    MPI_Type_size(datatype, &tySize);
    numBytes = tySize * count;
  }

  // use staged for big remote messages
  if (!is_colocated(comm, dest) && numBytes > (1 << 18) &&
      numBytes < (1 << 21)) {
    LOG_SPEW("MPI_Send: staged");
    return staged(numBytes, ARGS_MPI_Send);
  }

  // if all else fails, just do MPI_Send
  LOG_SPEW("MPI_Send: use library (fallthrough)");
  return libmpi.MPI_Send(ARGS_MPI_Send);
}
