/* TODO: anything to be learned from the MPITypes project?

   or loop optimizations? unimodular or polyhedral?

   or array access analysis?
*/

#pragma once

#include "packer.hpp"

#include "logging.hpp"
#include "strided_block.hpp"

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

Type traverse(MPI_Datatype datatype);
Type simplify(const Type &type);
StridedBlock to_strided_block(const Type &type);
std::unique_ptr<Packer> plan_pack(const StridedBlock &sb);

// release any resources associated with a datatype
void release(MPI_Datatype ty);

/*analyze some named types*/
void types_init();