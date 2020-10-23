/* TODO: anything to be learned from the MPITypes project?

   or loop optimizations? unimodular or polyhedral?

   or array access analysis?
*/

#pragma once

#include "packer.hpp"

#include <mpi.h>

#include <cstdint>
#include <map>
#include <memory>
#include <variant>
#include <vector>

struct DenseData {
  int64_t extent;

  bool operator==(const DenseData &rhs) const noexcept {
    return extent == rhs.extent;
  }

  std::string str() const noexcept {
    return std::string("DenseData{extent: ") + std::to_string(extent) + "}";
  }
};

struct VectorData {
  int64_t size;
  int64_t extent;

  int64_t count;
  int64_t blockLength;
  int64_t stride; // stride in bytes (hvector in MPI, not like vector)

  bool operator==(const VectorData &rhs) const noexcept {
    return size == rhs.size && extent == rhs.extent && count == rhs.count &&
           blockLength == rhs.blockLength && stride == rhs.stride;
  }

  std::string str() const noexcept {

    std::string s("VectorData{");
    s += "count:" + std::to_string(count);
    s += ",blockLength:" + std::to_string(blockLength);
    s += ",stride:" + std::to_string(stride);
    s += ",size:" + std::to_string(size);
    s += ",extent:" + std::to_string(extent);
    s += "}";
    return s;
  }
};

struct SubarrayData {
  std::vector<int64_t> elemSubsizes; // size of subarray in elements
  std::vector<int64_t> elemStarts;   // offset fo subarray in elements
  std::vector<int64_t> elemSizes;

  size_t ndims() const noexcept { return elemSubsizes.size(); }

  bool operator==(const SubarrayData &rhs) const noexcept {
    return elemSubsizes == rhs.elemSubsizes && elemStarts == rhs.elemStarts &&
           elemSizes == rhs.elemSizes;
  }

  void erase_dim(size_t i) {
    elemSubsizes.erase(elemSubsizes.begin() + i);
    elemStarts.erase(elemStarts.begin() + i);
    elemSizes.erase(elemSizes.begin() + i);
  }

  std::string str() const noexcept {

    auto as_string = [](const std::vector<int64_t> &v) -> std::string {
      std::string s("[");
      for (int i : v) {
        s += std::to_string(i) + " ";
      }
      s += "]";
      return s;
    };

    std::string s("SubarrayData{");
    s += "elemSubsizes:" + as_string(elemSubsizes);
    s += ",elemStarts:" + as_string(elemStarts);
    s += ",elemSizes:" + as_string(elemSizes);
    s += "}";
    return s;
  }
};

typedef std::variant<std::monostate, DenseData, VectorData, SubarrayData>
    TypeData;

/* a tree representing an MPI datatype
 */
class Type {
  std::vector<Type> children_;

public:
  TypeData data;

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

/*
 */
struct StridedBlock {
  StridedBlock() : blockLength(-1) {}
  int64_t blockLength;

  std::vector<int64_t> starts;
  std::vector<int64_t> counts;
  std::vector<int64_t> strides;

  size_t ndims() const noexcept { return starts.size(); }

  void add_dim(int64_t start, int64_t count, int64_t stride) {
    starts.push_back(start);
    counts.push_back(count);
    strides.push_back(stride);
  }

  bool operator==(const StridedBlock &rhs) const noexcept {
    return blockLength == rhs.blockLength && starts == rhs.starts &&
           counts == rhs.counts && strides == rhs.strides;
  }

  bool operator!=(const StridedBlock &rhs) const noexcept {
    return !(*this == rhs);
  }

  std::string str() const noexcept {

    auto as_string = [](const std::vector<int64_t> &v) -> std::string {
      std::string s("[");
      for (int i : v) {
        s += std::to_string(i) + " ";
      }
      s += "]";
      return s;
    };

    std::string s("StridedBlock{");
    s += "blockLength:" + std::to_string(blockLength);
    s += ",starts:" + as_string(starts);
    s += ",counts:" + as_string(counts);
    s += ",strides:" + as_string(strides);
    s += "}";
    return s;
  }
};

extern std::map<MPI_Datatype, std::shared_ptr<Packer>> packerCache;

Type traverse(MPI_Datatype datatype);

std::shared_ptr<Packer> plan_pack(Type &type);

StridedBlock to_strided_block(const Type &type);