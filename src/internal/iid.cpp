#include "iid.hpp"

#include <cassert>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::time_point<Clock, Duration> Time;

std::random_device rd;
/*extern*/ std::mt19937 g(rd());

inline double med(const std::vector<double> &s) {
  std::vector<double> sp = s;
  std::sort(sp.begin(), sp.end());
  if (sp.size() % 2) {
    return (sp[sp.size() / 2] + sp[sp.size() / 2 + 1]) / 2;
  } else {
    return sp[sp.size() / 2];
  }
}

inline bool eq(double a, double b) { return a == b; }

inline double get_excursion(const std::vector<double> &s) {
  double xBar = avg(s);
  double runSum = 0;
  double maxExc = -1;
  for (size_t i = 0; i < s.size(); ++i) {
    runSum += s[i];
    double exc = std::abs(double(runSum) - (i + 1) * xBar);
    maxExc = std::max(maxExc, exc);
  }
  assert(maxExc >= 0);
  return maxExc;
}

// number of directional runs.
inline double get_num_runs(const std::vector<double> &s) {

  size_t numRuns = 1;
  for (size_t i = 0; i < s.size() - 2; ++i) {
    bool spi = s[i] <= s[i + 1];
    bool spj = s[i + 1] <= s[i + 2];
    if (spi != spj) {
      ++numRuns;
    }
  }

  return numRuns;
}

inline double get_longest_run(const std::vector<double> &s) {

  int64_t currRun = 1, longestRun = 0;

  for (size_t i = 0; i < s.size() - 2; ++i) {
    bool spi = s[i] <= s[i + 1];
    bool spj = s[i + 1] <= s[i + 2];
    if (spi == spj) {
      ++currRun;
    } else {
      longestRun = std::max(currRun, longestRun);
      currRun = 1;
    }
  }

  return longestRun;
}

inline double get_num_inc_dec(const std::vector<double> &s) {

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

inline double get_num_runs_med(const std::vector<double> &s) {

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

inline double get_longest_run_med(const std::vector<double> &s) {

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

inline double get_average_collision(const std::vector<double> &s) {

  std::vector<double> c;

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

inline double get_max_collision(const std::vector<double> &s) {

  std::vector<double> c;

  int64_t i = 0;
  int64_t maxC = -1;
  while (i < int64_t(s.size())) {
    int64_t j;
    for (j = i + 1; j < int64_t(s.size()); ++j) {
      if (s[i] == s[j]) {
        maxC = std::max(maxC, j - i);
        break;
      }
    }
    i = j + 1;
  }

  return maxC;
}

struct Test {
  double (*fn)(const std::vector<double> &);
  const char *name;
};

Test tests[]{{get_excursion, "excursion"},
             {get_num_runs, "num_runs"},
             {get_longest_run, "longest_run"},
             {get_num_inc_dec, "num_inc_dec"},
             {get_num_runs_med, "num_runs_med"},
             {get_longest_run_med, "longest_run_med"},
             {get_average_collision, "average_collision"},
             {get_max_collision, "max_collision"}};

bool sp_800_90B(const std::vector<double> &s) {

#if 0
  for (size_t i = 0; i < 10; ++i) {
    std::cerr << s[i] << " ";
  }
  std::cerr << "\n";
  for (size_t i = 0; i < 10; ++i) {
    std::cerr << fixed[i] << " ";
  }
  std::cerr << "\n";
#endif

  constexpr int NUM_TESTS = sizeof(tests) / sizeof(tests[0]);

  // use this sort of strange construction so that we can reuse the
  // std::shuffle which is slow for large numbers of samples
  int64_t t[NUM_TESTS]{}, tp[NUM_TESTS][10000]{};
  for (int i = 0; i < NUM_TESTS; ++i) {
    //Time start = Clock::now();
    t[i] = tests[i].fn(s);
    //Time stop = Clock::now();
    //std::cerr << i << " " << Duration(stop - start).count() << "\n";
  }

  for (int j = 0; j < 10000; ++j) {
    std::vector<double> sp = s;
    std::shuffle(sp.begin(), sp.end(), g);
    for (int i = 0; i < NUM_TESTS; ++i) {
      tp[i][j] = tests[i].fn(sp);
#if 0
      if (1 == i && j < 10) {
        std::cerr << tp[i][j] << " ";
      }
      if (1 == i && j == 11) {
        std::cerr << "\n";
      }
#endif
    }
  }

  for (int i = 0; i < NUM_TESTS; ++i) {
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
#if 0
    std::cerr << C0 + C1 << " " << C0 << "\n";
#endif
    if (C0 + C1 <= 5) {
      std::cerr << "FAIL (high) " << tests[i].name << "\n";
      return false;
    }
    if (C0 >= 9995) {
      std::cerr << "FAIL (low) " << tests[i].name << "\n";
      return false;
    }
  }
  return true;
}
