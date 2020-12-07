#include "iid.hpp"

std::random_device rd;
/*extern*/ std::mt19937 g(rd());

inline double avg(const std::vector<int64_t> &s) {
  double a = 0;
  for (int64_t e : s) {
    a += e;
  }
  return a / s.size();
}

inline double med(const std::vector<int64_t> &s) {
  if (s.size() % 2) {
    return (s[s.size() / 2] + s[s.size() / 2 + 1]) / 2;
  } else {
    return s[s.size() / 2];
  }
}

inline int64_t get_excursion(const std::vector<int64_t> &s) {
  double xBar = avg(s);
  int64_t runSum = 0;
  double maxExc = -1;
  for (size_t i = 0; i < s.size(); ++i) {
    runSum += s[i];
    double exc = std::abs(runSum - (i + 1) * xBar);
    maxExc = std::max(maxExc, exc);
  }
  return maxExc;
}

inline int64_t get_num_runs(const std::vector<int64_t> &s) {

  std::vector<bool> sp(s.size() - 1);

  for (size_t i = 0; i < s.size() - 1; ++i) {
    if (s[i] > s[i + 1]) {
      sp[i] = false;
    } else {
      sp[i] = true;
    }
  }

  size_t numRuns = 1;
  for (size_t i = 0; i < sp.size() - 1; ++i) {
    if (sp[i] != sp[i + 1]) {
      ++numRuns;
    }
  }
  return numRuns;
}

inline int64_t get_longest_run(const std::vector<int64_t> &s) {

  std::vector<int> sp;

  for (size_t i = 0; i < s.size() - 1; ++i) {
    if (s[i] > s[i + 1]) {
      sp.push_back(-1);
    } else {
      sp.push_back(1);
    }
  }

  int64_t currRun = 1, longestRun = -1;
  for (size_t i = 0; i < sp.size() - 1; ++i) {
    if (sp[i] == sp[i + 1]) {
      ++currRun;
    } else {
      longestRun = std::max(currRun, longestRun);
      currRun = 1;
    }
  }
  return longestRun;
}

inline int64_t get_num_inc_dec(const std::vector<int64_t> &s) {

  int64_t ni = 0, nd = 0;

  for (size_t i = 0; i < s.size() - 1; ++i) {
    if (s[i] > s[i + 1]) {
      ++nd;
    } else {
      ++ni;
    }
  }

  return std::max(nd, ni);
}

inline int64_t get_num_runs_med(const std::vector<int64_t> &s) {

  double m = med(s);
  std::vector<int> sp;

  for (size_t i = 0; i < s.size() - 1; ++i) {
    if (s[i] < m) {
      sp.push_back(-1);
    } else {
      sp.push_back(1);
    }
  }

  size_t numRuns = 1;
  for (size_t i = 0; i < sp.size() - 1; ++i) {
    if (sp[i] != sp[i + 1]) {
      ++numRuns;
    }
  }
  return numRuns;
}

inline int64_t get_longest_run_med(const std::vector<int64_t> &s) {

  double m = med(s);
  std::vector<int> sp;

  for (size_t i = 0; i < s.size() - 1; ++i) {
    if (s[i] < m) {
      sp.push_back(-1);
    } else {
      sp.push_back(1);
    }
  }

  int64_t currRun = 1, longestRun = -1;
  for (size_t i = 0; i < sp.size() - 1; ++i) {
    if (sp[i] == sp[i + 1]) {
      ++currRun;
    } else {
      longestRun = std::max(currRun, longestRun);
      currRun = 1;
    }
  }
  return longestRun;
}

inline int64_t get_average_collision(const std::vector<int64_t> &s) {

  std::vector<int64_t> c;

  int64_t i = 0;
  while (i < int64_t(s.size())) {
    size_t j;
    for (j = i + 1; j < s.size(); ++j) {
      if (s[i] == s[j]) {
        c.push_back(j - i);
        break;
      }
    }
    i = j + 1;
  }

  return avg(c);
}

inline int64_t get_max_collision(const std::vector<int64_t> &s) {

  std::vector<int64_t> c;

  size_t length;
  int64_t start = s[0];
  int64_t i = 0;
  int64_t maxC = -1;
  while (i < s.size()) {
    int64_t j;
    for (j = i + 1; j < s.size(); ++j) {
      if (s[i] == s[j]) {
        maxC = std::max(maxC, j - i);
        break;
      }
    }
    i = j + 1;
  }

  return maxC;
}

bool sp_800_90B(const std::vector<double> &s) {

  std::vector<int64_t> fixed = to_fixed(s);
  for (size_t i = 0; i < 10; ++i) {
    std::cerr << s[i] << " ";
  }
  std::cerr << "\n";
  for (size_t i = 0; i < 10; ++i) {
    std::cerr << fixed[i] << " ";
  }
  std::cerr << "\n";

  // use this sort of strange construction so that we can reuse the
  // std::shuffle which is slow for large numbers of samples
  int64_t t[8]{}, tp[8][10000]{};
  t[0] = get_excursion(fixed);
  t[1] = get_num_runs(fixed);
  t[2] = get_longest_run(fixed);
  t[3] = get_num_inc_dec(fixed);
  t[4] = get_num_runs_med(fixed);
  t[5] = get_longest_run_med(fixed);
  t[6] = get_average_collision(fixed);
  t[7] = get_max_collision(fixed);
  for (int j = 0; j < 10000; ++j) {
    std::vector<int64_t> fixedp = fixed;
    std::shuffle(fixedp.begin(), fixedp.end(), g);

    tp[0][j] = get_excursion(fixedp);
    tp[1][j] = get_num_runs(fixedp);
    tp[2][j] = get_longest_run(fixedp);
    tp[3][j] = get_num_inc_dec(fixedp);
    tp[4][j] = get_num_runs_med(fixedp);
    tp[5][j] = get_longest_run_med(fixedp);
    tp[6][j] = get_average_collision(fixedp);
    tp[7][j] = get_max_collision(fixedp);
  }

  for (int i = 0; i < 8; ++i) {
    int64_t C0 = 0, C1 = 0;
    for (int j = 0; j < 10000; ++j) {
      if (tp[i][j] > t[i])
        ++C0;
      if (tp[i][j] == t[i])
        ++C1;
    }
    // original test statistic has a very high rank
    // original test statistic has a very low rank
    // false positive probability of 0.001
    std::cerr << C0 + C1 << " " << C0 << "\n";
    if (C0 + C1 <= 5 || C0 >= 9995) {
      return false;
    }
  }
  return true;
}