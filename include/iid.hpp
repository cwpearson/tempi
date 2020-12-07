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

int64_t get_excursion(const std::vector<int64_t> &s);
int64_t get_num_runs(const std::vector<int64_t> &s);
int64_t get_longest_run(const std::vector<int64_t> &s);
int64_t get_num_inc_dec(const std::vector<int64_t> &s);
int64_t get_num_runs_med(const std::vector<int64_t> &s);
int64_t get_longest_run_med(const std::vector<int64_t> &s);
int64_t get_average_collision(const std::vector<int64_t> &s);
int64_t get_max_collision(const std::vector<int64_t> &s);

typedef int64_t (*PermutationTest)(const std::vector<int64_t> &s);

template <PermutationTest FN>
bool permutation_test(const std::vector<int64_t> &s) {

  int C0 = 0, C1 = 0;

  const int64_t t = FN(s);
  for (int j = 0; j < 10000; ++j) {
    std::vector<int64_t> sp = s;
    std::shuffle(sp.begin(), sp.end(), g);
    const int64_t tp = FN(sp);

    if (tp > t)
      ++C0;
    if (tp == t)
      ++C1;
  }

  std::cerr << C0 + C1 << " " << C0 << "\n";
  // original test statistic has a very high rank
  // original test statistic has a very low rank
  // false positive probability of 0.001
  if (C0 + C1 <= 5 || C0 >= 9995) {
    return false;
  }

  return true;
}

bool sp_800_90B(const std::vector<double> &s);