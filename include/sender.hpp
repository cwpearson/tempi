#pragma once

#include "measure_system.hpp"
#include "packer.hpp"
#include "strided_block.hpp"
#include "symbols.hpp"

#include <mpi.h>

#include <memory>
#include <unordered_map>

/* interface for all senders */
class Sender {
public:
  virtual ~Sender() {}
  virtual int send(PARAMS_MPI_Send) = 0;
};
class Recver {
public:
  virtual ~Recver() {}
  virtual int recv(PARAMS_MPI_Recv) = 0;
};

/* always call the library send */
class SendRecvFallback : public Sender, public Recver {

public:
  virtual int send(PARAMS_MPI_Send) override;
  virtual int recv(PARAMS_MPI_Recv) override;

  static double model(const SystemPerformance &sp, bool colocated,
                      int64_t bytes);
};

class SendRecv1DStaged : public Sender, public Recver {
  int elemSize;

public:
  SendRecv1DStaged(const StridedBlock &sb) { elemSize = sb.counts[0]; }
  virtual int send(PARAMS_MPI_Send) override;
  virtual int recv(PARAMS_MPI_Recv) override;
  double model(const SystemPerformance &sp, bool colocated, int64_t bytes);
};

/*
 A sender for 1D data.
 Decides between staging on the host, or sending directly
*/
class SendRecv1D : public Sender, public Recver {
  int64_t elemSize;

  SendRecv1DStaged staged;

public:
  SendRecv1D(const StridedBlock &sb) : elemSize(sb.counts[0]), staged(sb) {}
  virtual int send(PARAMS_MPI_Send) override;
  virtual int recv(PARAMS_MPI_Recv) override;
};

class OneshotND : public Sender, public Recver {
  std::unique_ptr<Packer> packer_;

public:
  OneshotND(const StridedBlock &sb);
  virtual int send(PARAMS_MPI_Send) override;
  virtual int recv(PARAMS_MPI_Recv) override;
  double model(const SystemPerformance &sp, bool colocated, int64_t bytes,
               int64_t blockLength);
};

class DeviceND : public Sender, public Recver {
  std::unique_ptr<Packer> packer_;

public:
  DeviceND(const StridedBlock &sb);
  virtual int send(PARAMS_MPI_Send) override;
  virtual int recv(PARAMS_MPI_Recv) override;
  double model(const SystemPerformance &sp, bool colocated, int64_t bytes,
               int64_t blockLength);
};

/*not currently used*/
class StagedND : public Sender, public Recver {
  std::unique_ptr<Packer> packer_;

public:
  StagedND(const StridedBlock &sb);
  virtual int send(PARAMS_MPI_Send) override;
  virtual int recv(PARAMS_MPI_Recv) override;
  double model(const SystemPerformance &sp, bool colocated, int64_t bytes,
               int64_t blockLength);
};

class SendRecvND : public Sender, public Recver {
  OneshotND oneshot;
  DeviceND device;
  int64_t blockLength_;

  // argument pack to sender
  struct Args {
    bool colocated;
    int64_t bytes;
    struct Hasher { // unordered_map key
      size_t operator()(const Args &a) const noexcept {
        return std::hash<bool>()(a.colocated) ^ std::hash<int64_t>()(a.bytes);
      }
    };
    bool operator==(const Args &rhs) const { // unordered_map key
      return colocated == rhs.colocated && bytes == rhs.bytes;
    }
    bool operator<(const Args &rhs) const noexcept; // map key
  };

  // which sender to use
  enum class Method { DEVICE, ONESHOT };

  std::unordered_map<Args, Method, Args::Hasher> modelChoiceCache_;

public:
  SendRecvND(const StridedBlock &sb) : oneshot(sb), device(sb) {
    blockLength_ = sb.counts[0];
    blockLength_ = std::max(int64_t(1), blockLength_);
    blockLength_ = std::min(int64_t(512), blockLength_);
  }
  virtual int send(PARAMS_MPI_Send) override;
  virtual int recv(PARAMS_MPI_Recv) override;
};