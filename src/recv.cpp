#include "allocators.hpp"
#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "packer_cache.hpp"
#include "symbols.hpp"
#include "topology.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <vector>

static int recv_device(int device, Packer &packer, PARAMS_MPI_Recv) {
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

  // recv into temporary buffer
  int err = libmpi.MPI_Recv(packBuf, packedBytes, MPI_PACKED, source, tag, comm,
                            status);

  // unpack from temporary buffer
  int pos = 0;
  packer.unpack(packBuf, &pos, buf, count);

  // release temporary buffer
  LOG_SPEW("free intermediate recv buffer");
  deviceAllocator.deallocate(packBuf, packedBytes);
  return err;
}

/* recv data into pinned buffer and unpack into GPU */
static int recv_oneshot(int device, Packer &packer, PARAMS_MPI_Recv) {
  CUDA_RUNTIME(cudaSetDevice(device));

  // recv into device buffer
  int packedBytes;
  {
    int tySize;
    MPI_Type_size(datatype, &tySize);
    packedBytes = tySize * count;
  }
  void *packBuf = nullptr;
  packBuf = hostAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B device recv buffer");

  // recv into temporary buffer
  int err = libmpi.MPI_Recv(packBuf, packedBytes, MPI_PACKED, source, tag, comm,
                            status);

  // unpack from temporary buffer
  int pos = 0;
  packer.unpack(packBuf, &pos, buf, count);

  // release temporary buffer
  LOG_SPEW("free intermediate recv buffer");
  hostAllocator.deallocate(packBuf, packedBytes);
  return err;
}

static int staged(int numBytes, // pre-computed buffer size in bytes
                  PARAMS_MPI_Recv) {

  // reserve intermediate buffer
  void *hostBuf = hostAllocator.allocate(numBytes);

  // send to other device
  int err =
      libmpi.MPI_Recv(hostBuf, count, datatype, source, tag, comm, status);

  // copy to device
  CUDA_RUNTIME(cudaMemcpy(buf, hostBuf, numBytes, cudaMemcpyHostToDevice));

  // release temporary buffer
  hostAllocator.deallocate(hostBuf, numBytes);

  return err;
}

extern "C" int MPI_Recv(PARAMS_MPI_Recv) {
  if (environment::noTempi) {
    return libmpi.MPI_Recv(ARGS_MPI_Recv);
  }
  source = topology::library_rank(comm, source);

  // use library MPI for memory we can't reach on the device
  cudaPointerAttributes attr = {};
  CUDA_RUNTIME(cudaPointerGetAttributes(&attr, buf));
  if (nullptr == attr.devicePointer) {
    LOG_SPEW("MPI_Recv: use library (host memory)");
    return libmpi.MPI_Recv(ARGS_MPI_Recv);
  }

  // optimize packer
  auto pi = packerCache.find(datatype);
  if (packerCache.end() != pi) {

    switch (environment::datatype) {
    case DatatypeMethod::ONESHOT: {
      LOG_SPEW("MPI_Recv: oneshot");
      return recv_oneshot(attr.device, *(pi->second), ARGS_MPI_Recv);
    }
    case DatatypeMethod::DEVICE: {
      LOG_SPEW("MPI_Recv: device");
      return recv_device(attr.device, *(pi->second), ARGS_MPI_Recv);
    }
    default: {
      LOG_ERROR("unexpected DatatypeMethod");
      return MPI_ERR_UNKNOWN;
    }
    }
  }

  // message size
  int numBytes;
  {
    int tySize;
    MPI_Type_size(datatype, &tySize);
    numBytes = tySize * count;
  }

  // use staged for big remote messages
  if (!is_colocated(comm, source) && numBytes > (1 << 18) &&
      numBytes < (1 << 21)) {
    LOG_SPEW("MPI_Recv: staged");
    return staged(numBytes, ARGS_MPI_Recv);
  }

  // if all else fails, just call MPI_Recv
  LOG_SPEW("MPI_Recv: use library (fallthrough)");
  LOG_SPEW("count=" << count << " source=" << source << " tag=" << tag);
  return libmpi.MPI_Recv(ARGS_MPI_Recv);
}
