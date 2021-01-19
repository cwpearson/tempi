//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "sender.hpp"

#include "allocators.hpp"
#include "counters.hpp"
#include "packer_2d.hpp"
#include "packer_3d.hpp"
#include "topology.hpp"

#include <chrono>
typedef std::chrono::system_clock Clock;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::time_point<Clock, Duration> Time;

int SendRecvFallback::send(PARAMS_MPI_Send) {
  return libmpi.MPI_Send(ARGS_MPI_Send);
}
int SendRecvFallback::recv(PARAMS_MPI_Recv) {
  return libmpi.MPI_Recv(ARGS_MPI_Recv);
}

double SendRecvFallback::model(const SystemPerformance &sp, bool colocated,
                               int64_t bytes) {
  double send = interp_time(colocated ? sp.intraNodeGpuGpuPingpong
                                      : sp.interNodeGpuGpuPingpong,
                            bytes);
  return send;
}

int SendRecv1DStaged::send(PARAMS_MPI_Send) {
  int64_t numBytes = elemSize * count;
  void *hostBuf = hostAllocator.allocate(numBytes);
  CUDA_RUNTIME(cudaMemcpy(hostBuf, buf, numBytes, cudaMemcpyDeviceToHost));
  int err = libmpi.MPI_Send(hostBuf, count, datatype, dest, tag, comm);
  hostAllocator.deallocate(hostBuf, numBytes);
  return err;
}

int SendRecv1DStaged::recv(PARAMS_MPI_Recv) {
  int64_t numBytes = elemSize * count;
  void *hostBuf = hostAllocator.allocate(numBytes);
  int err =
      libmpi.MPI_Recv(hostBuf, count, datatype, source, tag, comm, status);
  CUDA_RUNTIME(cudaMemcpy(buf, hostBuf, numBytes, cudaMemcpyHostToDevice));
  hostAllocator.deallocate(hostBuf, numBytes);
  return err;
}

double SendRecv1DStaged::model(const SystemPerformance &sp, bool colocated,
                               int64_t bytes) {
  double d2h = interp_time(sp.d2h, bytes);
  double send = interp_time(colocated ? sp.intraNodeGpuGpuPingpong
                                      : sp.interNodeGpuGpuPingpong,
                            bytes);
  double h2d = interp_time(sp.h2d, bytes);
  return d2h + send + h2d;
}

int SendRecv1D::send(PARAMS_MPI_Send) {
  int64_t bytes = elemSize * count;
  bool colocated = is_colocated(comm, dest);
  double sm = staged.model(systemPerformance, colocated, bytes);
  double fm = SendRecvFallback::model(systemPerformance, colocated, bytes);
  LOG_SPEW("Sender1D::send: sm=" << sm << " fm=" << fm);
  if (sm < fm) {
    return staged.send(ARGS_MPI_Send);
  } else {
    return libmpi.MPI_Send(ARGS_MPI_Send);
  }
}

int SendRecv1D::recv(PARAMS_MPI_Recv) {
  int64_t bytes = elemSize * count;
  bool colocated = is_colocated(comm, source);
  double sm = staged.model(systemPerformance, colocated, bytes);
  double fm = SendRecvFallback::model(systemPerformance, colocated, bytes);
  if (sm < fm) {
    return staged.recv(ARGS_MPI_Recv);
  } else {
    return libmpi.MPI_Recv(ARGS_MPI_Recv);
  }
}

OneshotND::OneshotND(const StridedBlock &sb) {
  if (2 == sb.ndims()) {
    packer_ = std::make_unique<Packer2D>(sb.start_, sb.counts[0], sb.counts[1],
                                         sb.strides[1], sb.extent_);
  } else if (3 == sb.ndims()) {
    packer_ =
        std::make_unique<Packer3D>(sb.start_, sb.counts[0], sb.counts[1],
                                   sb.strides[1], sb.counts[2], sb.strides[2], sb.extent_);
  } else {
    LOG_FATAL("unhandled number of dimensions");
  }
}

int OneshotND::send(PARAMS_MPI_Send) {
  int packedBytes;
  MPI_Pack_size(count, datatype, comm, &packedBytes);
  void *packBuf = nullptr;
  packBuf = hostAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B host send buffer");
  int pos = 0;
  packer_->pack(packBuf, &pos, buf, count);
  int err = libmpi.MPI_Send(packBuf, packedBytes, MPI_PACKED, dest, tag, comm);
  hostAllocator.deallocate(packBuf, packedBytes);
  return err;
}

int OneshotND::recv(PARAMS_MPI_Recv) {
  int packedBytes;
  MPI_Pack_size(count, datatype, comm, &packedBytes);
  void *packBuf = nullptr;
  packBuf = hostAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B host recv buffer");
  int err = libmpi.MPI_Recv(packBuf, packedBytes, MPI_PACKED, source, tag, comm,
                            status);
  int pos = 0;
  packer_->unpack(packBuf, &pos, buf, count);
  hostAllocator.deallocate(packBuf, packedBytes);
  return err;
}

double OneshotND::model(const SystemPerformance &sp, bool colocated,
                        int64_t bytes, int64_t blockLength) {
  double d2h = interp_2d(sp.packHost, bytes, blockLength);
  double send = interp_time(colocated ? sp.intraNodeCpuCpuPingpong
                                      : sp.interNodeCpuCpuPingpong,
                            bytes);
  // FXIME: unpack not pack
  double h2d = interp_2d(sp.packHost, bytes, blockLength);
  return d2h + send + h2d;
}

DeviceND::DeviceND(const StridedBlock &sb) {
  if (2 == sb.ndims()) {
    packer_ = std::make_unique<Packer2D>(sb.start_, sb.counts[0], sb.counts[1],
                                         sb.strides[1], sb.extent_);
  } else if (3 == sb.ndims()) {
    packer_ =
        std::make_unique<Packer3D>(sb.start_, sb.counts[0], sb.counts[1],
                                   sb.strides[1], sb.counts[2], sb.strides[2], sb.extent_);
  } else {
    LOG_FATAL("unhandled number of dimensions");
  }
}

int DeviceND::send(PARAMS_MPI_Send) {
  int packedBytes;
  MPI_Pack_size(count, datatype, comm, &packedBytes);
  void *packBuf = nullptr;
  packBuf = deviceAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B device send buffer");
  int pos = 0;
  packer_->pack(packBuf, &pos, buf, count);
  int err = libmpi.MPI_Send(packBuf, packedBytes, MPI_PACKED, dest, tag, comm);
  deviceAllocator.deallocate(packBuf, packedBytes);
  return err;
}

int DeviceND::recv(PARAMS_MPI_Recv) {
  int packedBytes;
  MPI_Pack_size(count, datatype, comm, &packedBytes);
  void *packBuf = nullptr;
  packBuf = deviceAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B device send buffer");
  int err = libmpi.MPI_Recv(packBuf, packedBytes, MPI_PACKED, source, tag, comm,
                            status);
  int pos = 0;
  packer_->unpack(packBuf, &pos, buf, count);
  deviceAllocator.deallocate(packBuf, packedBytes);
  return err;
}

double DeviceND::model(const SystemPerformance &sp, bool colocated,
                       int64_t bytes, int64_t blockLength) {
  double pack = interp_2d(sp.packDevice, bytes, blockLength);
  double send = interp_time(colocated ? sp.intraNodeGpuGpuPingpong
                                      : sp.interNodeGpuGpuPingpong,
                            bytes);
  // FXIME: unpack not pack
  double unpack = interp_2d(sp.packDevice, bytes, blockLength);
  return pack + send + unpack;
}

StagedND::StagedND(const StridedBlock &sb) {
  if (2 == sb.ndims()) {
    packer_ = std::make_unique<Packer2D>(sb.start_, sb.counts[0], sb.counts[1],
                                         sb.strides[1], sb.extent_);
  } else if (3 == sb.ndims()) {
    packer_ =
        std::make_unique<Packer3D>(sb.start_, sb.counts[0], sb.counts[1],
                                   sb.strides[1], sb.counts[2], sb.strides[2], sb.extent_);
  } else {
    LOG_FATAL("unhandled number of dimensions");
  }
}

int StagedND::send(PARAMS_MPI_Send) {
  int packedBytes;
  MPI_Pack_size(count, datatype, comm, &packedBytes);
  void *packBuf = deviceAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B device packing buffer");
  int pos = 0;
  packer_->pack_async(packBuf, &pos, buf, count);
  void *hostBuf = hostAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B host send buffer");
  CUDA_RUNTIME(
      cudaMemcpy(hostBuf, packBuf, packedBytes, cudaMemcpyDeviceToHost));
  int err = libmpi.MPI_Send(hostBuf, count, datatype, dest, tag, comm);
  hostAllocator.deallocate(hostBuf, packedBytes);
  deviceAllocator.deallocate(packBuf, packedBytes);
  return err;
}

int StagedND::recv(PARAMS_MPI_Recv) {
  int packedBytes;
  MPI_Pack_size(count, datatype, comm, &packedBytes);
  void *packBuf = nullptr;
  packBuf = deviceAllocator.allocate(packedBytes);
  LOG_SPEW("allocate " << packedBytes << "B device recv buffer");
  int err = libmpi.MPI_Recv(packBuf, packedBytes, MPI_PACKED, source, tag, comm,
                            status);
  int pos = 0;
  packer_->unpack(packBuf, &pos, buf, count);
  LOG_SPEW("free intermediate recv buffer");
  deviceAllocator.deallocate(packBuf, packedBytes);
  return err;
}

double StagedND::model(const SystemPerformance &sp, bool colocated,
                       int64_t bytes, int64_t blockLength) {
  double pack = interp_2d(sp.packDevice, bytes, blockLength);
  double d2h = interp_time(sp.d2h, bytes);
  double send = interp_time(colocated ? sp.intraNodeCpuCpuPingpong
                                      : sp.interNodeCpuCpuPingpong,
                            bytes);
  double h2d = interp_time(sp.h2d, bytes);
  // FXIME: unpack not pack
  double unpack = interp_2d(sp.packDevice, bytes, blockLength);
  return pack + d2h + send + h2d + unpack;
}

bool SendRecvND::Args::operator<(const Args &rhs) const noexcept {
  if (colocated < rhs.colocated) {
    return true;
  } else {
    return bytes < rhs.bytes;
  }
}

int SendRecvND::send(PARAMS_MPI_Send) {
#ifdef TEMPI_ENABLE_COUNTERS
  double start = MPI_Wtime();
#endif
  int bytes;
  MPI_Pack_size(count, datatype, comm, &bytes);
  bool colocated = is_colocated(comm, dest);
  Args args{.colocated = colocated, .bytes = bytes};
  auto it = modelChoiceCache_.find(args);
  Method method;
  if (modelChoiceCache_.end() == it) {
    TEMPI_COUNTER_OP(modeling, CACHE_MISS, ++);
    double o = oneshot.model(systemPerformance, colocated, bytes, blockLength_);
    double d = device.model(systemPerformance, colocated, bytes, blockLength_);
    if (o < d) {
      method = Method::ONESHOT;
    } else {
      method = Method::DEVICE;
    }
    modelChoiceCache_[args] = method;
  } else {
    TEMPI_COUNTER_OP(modeling, CACHE_HIT, ++);
    method = it->second;
  }
  TEMPI_COUNTER_OP(modeling, WALL_TIME, += MPI_Wtime() - start);

  switch (method) {
  case Method::DEVICE:
    return device.send(ARGS_MPI_Send);
  case Method::ONESHOT:
    return oneshot.send(ARGS_MPI_Send);
  default:
    LOG_FATAL("unexpected send method");
  }
}

int SendRecvND::recv(PARAMS_MPI_Recv) {
#ifdef TEMPI_ENABLE_COUNTERS
  double start = MPI_Wtime();
#endif
  int bytes;
  MPI_Pack_size(count, datatype, comm, &bytes);
  bool colocated = is_colocated(comm, source);
  Args args{.colocated = colocated, .bytes = bytes};
  auto it = modelChoiceCache_.find(args);
  Method method;
  if (modelChoiceCache_.end() == it) {
    TEMPI_COUNTER_OP(modeling, CACHE_MISS, ++);
    double o = oneshot.model(systemPerformance, colocated, bytes, blockLength_);
    double d = device.model(systemPerformance, colocated, bytes, blockLength_);
    if (o < d) {
      method = Method::ONESHOT;
    } else {
      method = Method::DEVICE;
    }
    modelChoiceCache_[args] = method;
  } else {
    TEMPI_COUNTER_OP(modeling, CACHE_HIT, ++);
    method = it->second;
  }
  TEMPI_COUNTER_OP(modeling, WALL_TIME, += MPI_Wtime() - start);

  switch (method) {
  case Method::DEVICE:
    return device.recv(ARGS_MPI_Recv);
  case Method::ONESHOT:
    return oneshot.recv(ARGS_MPI_Recv);
  default:
    LOG_FATAL("unexpected recv method");
  }
}