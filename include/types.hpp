/* TODO: anything to be learned from the MPITypes project?

   or loop optimizations? unimodular or polyhedral?

   or array access analysis?
*/

#pragma once

#include "packer.hpp"

#include "logging.hpp"

#include <mpi.h>

#include <cstdint>
#include <map>
#include <memory>
#include <variant>
#include <vector>

struct DenseData {
  int64_t off;
  int64_t extent;

  bool operator==(const DenseData &rhs) const noexcept {
    return extent == rhs.extent;
  }

  std::string str() const noexcept {
    std::string s("DenseData{");
    s += "off:" + std::to_string(off);
    s += ",extent:" + std::to_string(extent);
    s += "}";
    return s;
  }
};

struct StreamData {
  int64_t off;    // offset (B) of the first element
  int64_t stride; // stride (B) between element starts in the stream
  int64_t count;  // number of elements in the stream

  bool operator==(const StreamData &rhs) const noexcept {
    return off == rhs.off && stride == rhs.stride && count == rhs.count &&
           count;
  }

  std::string str() const noexcept {

    std::string s("StreamData{");
    s += "off:" + std::to_string(off);
    s += ",count:" + std::to_string(count);
    s += ",stride:" + std::to_string(stride);
    s += "}";
    return s;
  }
};

typedef std::variant<std::monostate, DenseData, StreamData> TypeData;

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

  void str_helper(std::string &s, int indent) const {

    for (int i = 0; i < indent; ++i) {
      s += " ";
    }

    std::visit(
        [&](auto arg) {
          using T = std::decay_t<decltype(arg)>;
          if constexpr (std::is_same_v<T, std::monostate>) {
            s += "monostate";
          } else {

            s += arg.str();
          }
        },
        data);
    s += "\n";

    for (const Type &child : children_) {
      child.str_helper(s, indent + 1);
    }
  }
  std::string str() const {
    std::string s;
    str_helper(s, 0);
    if (s.back() == '\n') {
      s.erase(s.size() - 1);
    }
    return s;
  }
};

/*
 */
struct StridedBlock {
  StridedBlock() : start_(0) {}

  /* each dimension is described by a start/count/stride

     start: byte offset before elements in that dimension start
     count: number of elements in the dimension
     stride: (B) between start of each element

     so, the first byte is at the sum of the start of the dimensions:
     start[i-1] before elems dimension i-1 begins, then start [i-2]
     before the first element of i-2, etc

     so, instead of tracking start for each dimension, we track the
     total offset that the first byte starts at
  */
  int64_t start_;
  std::vector<int64_t> counts;
  std::vector<int64_t> strides;

  size_t ndims() const noexcept { return counts.size(); }

  void add_dim(int64_t start, int64_t count, int64_t stride) {
    start_ += start;
    counts.push_back(count);
    strides.push_back(stride);
  }

  bool operator==(const StridedBlock &rhs) const noexcept {
    return start_ == rhs.start_ && counts == rhs.counts &&
           strides == rhs.strides;
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
    s += "start:" + std::to_string(start_);
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

// release any resources associated with a datatype
void release(MPI_Datatype ty);