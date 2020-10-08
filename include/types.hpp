/* TODO: anything to be learned from the MPITypes project?

   or loop optimizations? unimodular or polyhedral?

   or array access analysis?
*/

#pragma once

#include "packer.hpp"

#include <mpi.h>

#include <map>
#include <memory>
#include <vector>

struct Vector {

  Vector()
      : count(-1), blockLength(-1), blockStride(-1), elemLength(-1),
        elemStride(-1) {}

  int64_t count;
  int64_t blockLength; // bytes
  int64_t blockStride; // bytes

  // supporting info used to compute the above when unknown
  int64_t elemLength; // block length in child elements
  int64_t elemStride; // block stride in child elements

  bool operator==(const Vector &rhs) const {
    return count == rhs.count && blockLength == rhs.blockLength &&
           elemStride == rhs.elemStride && blockStride == rhs.blockStride &&
           elemLength == rhs.elemLength && elemStride == rhs.elemStride;
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