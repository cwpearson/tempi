//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cstdint>
#include <string>
#include <vector>

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