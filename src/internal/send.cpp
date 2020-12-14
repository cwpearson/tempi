/*

implementations of send, and deciding what implementations to use

*/

#include "send.hpp"

#include "allocators.hpp"
#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "measure_system.hpp"
#include "packer_cache.hpp"
#include "topology.hpp"

#include <optional>

/* pack data into GPU buffer and send */
int send::pack_device_send(int device, Packer &packer, PARAMS_MPI_Send) {
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
int send::pack_host_send(int device, Packer &packer, PARAMS_MPI_Send) {
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

int send::staged(int numBytes, // pre-computed buffer size in bytes
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

/* pack data into device buffer, stage, and send
   overlap the pack with host allocation and initiating d2h
*/
int send::pack_device_stage(int device, Packer &packer, PARAMS_MPI_Send) {
  CUDA_RUNTIME(cudaSetDevice(device));

  // reserve device buffer
  int packedBytes;
  {
    int size;
    MPI_Pack_size(count, datatype, comm, &size);
    packedBytes = size;
  }
  void *packBuf = deviceAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B device packing buffer");

  // pack into device buffer
  int pos = 0;
  packer.pack_async(packBuf, &pos, buf, count);

  // reserve intermediate buffer
  void *hostBuf = hostAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B host send buffer");

  // copy to host
  CUDA_RUNTIME(
      cudaMemcpy(hostBuf, packBuf, packedBytes, cudaMemcpyDeviceToHost));

  // send to other device
  int err = libmpi.MPI_Send(hostBuf, count, datatype, dest, tag, comm);

  // release temporary buffers
  hostAllocator.deallocate(hostBuf, packedBytes);
  deviceAllocator.deallocate(packBuf, packedBytes);

  return err;
}

std::optional<double> guess_pack_device_send(const SystemPerformance &sp,
                                             int64_t bytes) {
  double pack = 0;
  double send = interp_time(sp.interNodeGpuGpuPingpong, bytes);
  double unpack = 0;
  return pack + send + unpack;
}

std::optional<double> guess_pack_device_stage(const SystemPerformance &sp,
                                              int64_t bytes) {
  double pack = 0;
  double d2h = interp_time(sp.d2h, bytes);
  double send = interp_time(sp.interNodeCpuCpuPingpong, bytes);
  double h2d = interp_time(sp.h2d, bytes);
  double unpack = 0;
  return pack + d2h + send + h2d + unpack;
}

std::optional<double> guess_pack_host_send(const SystemPerformance &sp,
                                           int64_t bytes) {
  double pack = 0;
  double send = interp_time(sp.interNodeCpuCpuPingpong, bytes);
  double unpack = 0;
  return pack + send + unpack;
}

int send::impl(PARAMS_MPI_Send) {
  LOG_SPEW("in send::impl");

  dest = topology::library_rank(comm, dest);

  // use library MPI for memory we can't reach on the device
  cudaPointerAttributes attr = {};
  CUDA_RUNTIME(cudaPointerGetAttributes(&attr, buf));
  if (nullptr == attr.devicePointer) {
    LOG_SPEW("send::impl: use library (host memory)");
    return libmpi.MPI_Send(ARGS_MPI_Send);
  }

  /* if we have a fast packer, decide between
   1) pack into device memory send
   2) pack into device memory and stage
   3) pack into host memory and send
  */

  // optimize packer
  auto pi = packerCache.find(datatype);
  if (packerCache.end() != pi) {

    switch (environment::datatype) {
    case DatatypeMethod::AUTO: {
      // message size
      int size;
      MPI_Pack_size(count, datatype, comm, &size);

      std::optional<double> pdSend =
          guess_pack_device_send(systemPerformance, size);
      std::optional<double> pdStage =
          guess_pack_device_stage(systemPerformance, size);
      std::optional<double> phSend =
          guess_pack_host_send(systemPerformance, size);

      if (pdStage <= pdSend && pdStage <= phSend) {
        return pack_device_stage(attr.device, *(pi->second), ARGS_MPI_Send);
      } else if (pdSend <= pdStage && pdSend <= phSend) {
        return pack_device_send(attr.device, *(pi->second), ARGS_MPI_Send);
      } else {
        return pack_host_send(attr.device, *(pi->second), ARGS_MPI_Send);
      }
    }
    case DatatypeMethod::ONESHOT: {
      LOG_SPEW("send::impl: send::pack_host_send");
      return send::pack_host_send(attr.device, *(pi->second), ARGS_MPI_Send);
    }
    case DatatypeMethod::DEVICE: {
      LOG_SPEW("send::impl: send::pack_device_send");
      return send::pack_device_send(attr.device, *(pi->second), ARGS_MPI_Send);
    }

    default: {
      LOG_ERROR("unexpected DatatypeMethod");
      return MPI_ERR_UNKNOWN;
    }
    }
  } else {
    LOG_SPEW("send::impl: no packer for " << uintptr_t(datatype));
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
    LOG_SPEW("send::impl: staged");
    return send::staged(numBytes, ARGS_MPI_Send);
  }

  // if all else fails, just do MPI_Send
  LOG_SPEW("send::impl: use library (fallthrough)");
  return libmpi.MPI_Send(ARGS_MPI_Send);
}