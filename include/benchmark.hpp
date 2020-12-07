#pragma once

#include <cstdint>

class Benchmark {

    protected:

    constexpr static int64_t minIters = 7; // empirically determined
    constexpr static double maxTrialTime = 0.04;
    constexpr static int64_t maxWarmupIter = 5;
    constexpr static double maxWarmupSecs = 0.01;
    constexpr static int64_t maxTrials = 10;

public:
  struct Result {
    int64_t nTrials; // number of trial runs
    int64_t nIters; // number of iterations in the final run
    double trimean;
  };

  struct IterResult {
      double time;
  };

  // should be overridden to take a single sample and return it
  virtual IterResult run_iter() = 0;


  Result run();
};