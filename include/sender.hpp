#pragma once

#include "measure_system.hpp"
#include "packer.hpp"
#include "strided_block.hpp"
#include "symbols.hpp"

#include <mpi.h>

#include <memory>

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

public:
  SendRecvND(const StridedBlock &sb) : oneshot(sb), device(sb) {
    blockLength_ = sb.counts[0];
    blockLength_ = std::max(int64_t(1), blockLength_);
    blockLength_ = std::min(int64_t(512), blockLength_);
  }
  virtual int send(PARAMS_MPI_Send) override;
  virtual int recv(PARAMS_MPI_Recv) override;
};