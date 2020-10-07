#pragma once

#include "packer.hpp"

#include <mpi.h>

#include <map>
#include <memory>
#include <vector>

struct Vector {

  Vector() : count(-1), blockLength(-1), elemStride(-1), byteStride(-1) {}

  int64_t count;
  int64_t blockLength; // bytes
  int64_t elemStride;  // bytes
  int64_t byteStride;

  bool operator==(const Vector &rhs) const {
    return count == rhs.count && blockLength == rhs.blockLength &&
           elemStride == rhs.elemStride && byteStride == rhs.byteStride;
  }
};

/* a tree representing an MPI datatype
 */
class Type {

  std::vector<Vector> levels_;

public:
  std::vector<Vector> &levels() { return levels_; }
  const std::vector<Vector> &levels() const { return levels_; }
  int num_levels() const { return levels_.size(); }

  static Type unknown() { return Type(); }

  bool operator==(const Type &rhs) const { return levels_ == rhs.levels_; }
  bool operator!=(const Type &rhs) const { return !(*this == rhs); }
};

extern std::map<MPI_Datatype, std::shared_ptr<Packer>> packerCache;

Type traverse(MPI_Datatype datatype);

std::shared_ptr<Packer> plan_pack(Type &type);