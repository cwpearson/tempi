/* kept out of types.cpp so that benchmark code can use the typeCache directly
 * without bring in c++17 for std::variant in TypeData
 */

#pragma once

#include "packer.hpp"
#include "sender.hpp"
#include "strided_block.hpp"

#include <mpi.h>

#include <map>
#include <memory>

struct TypeRecord {
  std::unique_ptr<Packer> packer;
  StridedBlock desc;
  std::unique_ptr<Sender> sender;
  std::unique_ptr<Recver> recver;
  int64_t mpiPackSize; // result of MPI_Pack_size
};

extern std::map<MPI_Datatype, TypeRecord> typeCache;