/* TODO: anything to be learned from the MPITypes project?

   or loop optimizations? unimodular or polyhedral?

   or array access analysis?
*/

#pragma once

#include "packer.hpp"

#include <mpi.h>

#include <map>
#include <memory>
#include <variant>
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

  bool is_contiguous() const noexcept {
    bool oneBlock = (1 == count);
    bool packedBytes =
        (blockLength >= 0 && blockStride >= 0 && blockLength == blockStride);
    bool packedElems =
        (elemLength >= 0 && elemStride >= 0 && elemLength == elemStride);
    return oneBlock || packedBytes || packedElems;
  }

  bool operator==(const Vector &rhs) const {
    return count == rhs.count && blockLength == rhs.blockLength &&
           elemStride == rhs.elemStride && blockStride == rhs.blockStride &&
           elemLength == rhs.elemLength && elemStride == rhs.elemStride;
  }
};

struct VectorData {
  VectorData()
      : count(-1), byteLength(-1), elemLength(-1), byteStride(-1),
        elemStride(-1) {}
  int count;
  int byteLength;
  int elemLength;
  int byteStride;
  int elemStride;

  bool operator==(const VectorData &rhs) const noexcept {
    return count == rhs.count && byteLength == rhs.byteLength &&
           elemLength == rhs.elemLength && byteStride == rhs.byteStride &&
           elemStride == rhs.elemStride;
  }

  std::string str() const noexcept {
    std::string s("VectorData{count:");
    s += std::to_string(count);
    s += " byteLength:" + std::to_string(byteLength);
    s += " byteStride:" + std::to_string(byteStride);
    s += " elemLength:" + std::to_string(elemLength);
    s += " elemStride:" + std::to_string(elemStride);
    s += "}";
    return s;
  }
};

struct SubarrayData {

  std::vector<int> elemSizes;
  std::vector<int> elemSubsizes;
  std::vector<int> elemStarts;

  // stride between the start of each dimension
  std::vector<int> byteStrides;

  // the length of the base element in bytes
  int byteLength;

  size_t ndims() const noexcept { return elemSizes.size(); }

  bool operator==(const SubarrayData &rhs) const noexcept {
    return elemSizes == rhs.elemSizes && elemSubsizes == rhs.elemSubsizes &&
           elemStarts == rhs.elemStarts && byteStrides == rhs.byteStrides &&
           byteLength == rhs.byteLength;
  }

  std::string str() const noexcept {

    auto as_string = [](const std::vector<int> &v) -> std::string {
      std::string s("[");
      for (int i : v) {
        s += std::to_string(i) + " ";
      }
      s += "]";
      return s;
    };

    std::string s("SubarrayData{");
    s += "byteLength:" + std::to_string(byteLength);
    s += ",elemSizes:" + as_string(elemSizes);
    s += ",elemSubsizes:" + as_string(elemSubsizes);
    s += ",elemStarts:" + as_string(elemStarts);
    s += ",byteStrides:" + as_string(byteStrides);
    s += "}";
    return s;
  }
};

/* a tree representing an MPI datatype
 */
class Type {
  std::vector<Type> children_;

public:
  std::variant<std::monostate, VectorData, SubarrayData> data;

  std::vector<Type> &children() { return children_; }
  const std::vector<Type> &children() const { return children_; }
  size_t height() const noexcept {
    if (children_.empty()) {
      return 0;
    } else {
      size_t tmp = 0;
      for (const Type &child : children_) {
        tmp = std::max(tmp, child.height());
      }
      return tmp + 1;
    }
  }

  bool operator==(const Type &rhs) const noexcept {
    return data == rhs.data && children_ == rhs.children_;
  }
  bool operator!=(const Type &rhs) const noexcept { return !(*this == rhs); }

  bool operator!() const noexcept { return *this == Type(); }

  static Type from_mpi_datatype(MPI_Datatype datatype);
};

extern std::map<MPI_Datatype, std::shared_ptr<Packer>> packerCache;

Type traverse(MPI_Datatype datatype);

std::shared_ptr<Packer> plan_pack(Type &type);