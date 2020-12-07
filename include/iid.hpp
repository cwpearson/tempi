/*
https://csrc.nist.gov/csrc/media/events/random-bit-generation-workshop-2016/documents/presentations/sessionii-3-meltem-sonmez-turan-presentation.pdf
*/

#pragma once

#include "logging.hpp"

#include <algorithm>
#include <random>
#include <vector>

extern std::mt19937 g;

inline std::vector<int64_t> to_fixed(const std::vector<double> &s) {
  double minVal = INFINITY;
  double maxVal = -1 * INFINITY;

  for (double d : s) {
    minVal = std::min(minVal, d);
    maxVal = std::max(maxVal, d);
  }

  // map [minVal, maxVal] to [0, 2^32-1]
  double sf = 4294967296.0 / (maxVal - minVal);
  std::vector<int64_t> ret;

  for (double d : s) {
    ret.push_back((d - minVal) * sf);
  }

  return ret;
}

bool sp_800_90B(const std::vector<double> &s);