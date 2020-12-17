/*

implementations of send, and deciding what implementations to use

*/

#include "send.hpp"

#include "allocators.hpp"
#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "measure_system.hpp"
#include "topology.hpp"
#include "type_cache.hpp"

#include <optional>
#if TEMPI_OUTPUT_LEVEL >= 4
#include <sstream>
#endif

#if 0
typedef int (*PackSend)(int device, Packer &packer, PARAMS_MPI_Send);
typedef int (*ContiguousSend)(int numBytes, // pre-computed buffer size in bytes
                              PARAMS_MPI_Send);

std::optional<double> guess_pack_device_send(const SystemPerformance &sp,
                                             const TypeRecord &tr) {

  size_t bytes = tr.size;
  double pack = 0;
  double send = interp_time(sp.intraNodeGpuGpuPingpong, bytes);
  double unpack = 0;
  return pack + send + unpack;
}

std::optional<double> guess_pack_device_stage(const SystemPerformance &sp,
                                              const TypeRecord &tr) {
  size_t bytes = tr.size;
  double pack = 0;
  double d2h = interp_time(sp.d2h, bytes);
  double send = interp_time(sp.intraNodeCpuCpuPingpong, bytes);
  double h2d = interp_time(sp.h2d, bytes);
  double unpack = 0;
  return pack + d2h + send + h2d + unpack;
}

std::optional<double> guess_pack_host_send(const SystemPerformance &sp,
                                           const TypeRecord &tr) {
  size_t bytes = tr.size;
  double pack = 0;
  double send = interp_time(sp.intraNodeCpuCpuPingpong, bytes);
  double unpack = 0;
  return pack + send + unpack;
}

std::optional<double> guess_contiguous_staged(const SystemPerformance &sp,
                                              const TypeRecord &tr) {
  size_t bytes = tr.size;
  double d2h = interp_time(sp.d2h, bytes);
  double send = interp_time(sp.intraNodeCpuCpuPingpong, bytes);
  double h2d = interp_time(sp.h2d, bytes);
  return d2h + send + h2d;
}

std::optional<double> guess_contiguous_send(const SystemPerformance &sp,
                                            const TypeRecord &tr) {
  size_t bytes = tr.size;
  double send = interp_time(sp.intraNodeGpuGpuPingpong, bytes);
  return send;
}

ContiguousSend select_contiguous(const SystemPerformance &sp,
                                 const TypeRecord &tr) {

  std::optional<double> staged = guess_contiguous_staged(sp, tr);
  std::optional<double> send = guess_contiguous_send(sp, tr);

  if (staged < send) {
    return send::staged;
  } else {
    return nullptr;
  }
}
PackSend select_packed(const SystemPerformance &sp, const TypeRecord &tr) {

  std::optional<double> pdSend = guess_pack_device_send(systemPerformance, tr);
  std::optional<double> pdStage =
      guess_pack_device_stage(systemPerformance, tr);
  std::optional<double> phSend = guess_pack_host_send(systemPerformance, tr);

#if TEMPI_OUTPUT_LEVEL >= 4
  std::stringstream ss;
  ss << "device_send=";
  if (pdSend) {
    ss << *pdSend;
  } else {
    ss << "{}";
  }
  ss << " ";
  ss << "device_stage=";
  if (pdStage) {
    ss << *pdStage;
  } else {
    ss << "{}";
  }
  ss << " ";
  ss << "host_send=";
  if (phSend) {
    ss << *phSend;
  } else {
    ss << "{}";
  }
  LOG_SPEW(ss.str());
#endif

  if (pdStage <= pdSend && pdStage <= phSend) {
    LOG_SPEW("selected pack_device_stage");
    return send::pack_device_stage;
  } else if (pdSend <= pdStage && pdSend <= phSend) {
    LOG_SPEW("selected pack_device_send");
    return send::pack_device_send;
  } else {
    LOG_SPEW("selected pack_host_send");
    return send::pack_host_send;
  }
}
#endif

#if 0
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
  MPI_Pack_size(count, datatype, comm, &packedBytes);

  void *packBuf = nullptr;
  packBuf = hostAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B host send buffer");

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
#endif

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

  auto pi = typeCache.find(datatype);

  // if sender is found
  if (typeCache.end() != pi && pi->second.sender) {
    LOG_SPEW("send::impl: cached Sender");
    return pi->second.sender->send(ARGS_MPI_Send);
  }

#if 0
  // if packer is found
  if (typeCache.end() != pi && pi->second.packer) {

    switch (environment::datatype) {
    case DatatypeMethod::ONESHOT: {
      LOG_SPEW("send::impl: send::pack_host_send");
      return send::pack_host_send(attr.device, *(pi->second.packer),
                                  ARGS_MPI_Send);
    }
    case DatatypeMethod::DEVICE: {
      LOG_SPEW("send::impl: send::pack_device_send");
      return send::pack_device_send(attr.device, *(pi->second.packer),
                                    ARGS_MPI_Send);
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
#endif

  // if all else fails, just do MPI_Send
  LOG_SPEW("send::impl: use library (fallthrough)");
  return libmpi.MPI_Send(ARGS_MPI_Send);
}