//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cstdlib>
#include <vector>

class Statistics {
private:
  std::vector<double> x;

public:
  void clear();
  void insert(double d);
  double avg() const;
  double min() const;
  double max() const;
  size_t count() const noexcept;
  double trimean();
  double med();
  double stddev() const;
  const std::vector<double> &raw() const { return x; }
};