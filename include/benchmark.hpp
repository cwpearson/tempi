#pragma once

#include <cstdint>

#include <mpi.h>

class Benchmark {

protected:
  // minimum guaranteed warmup iterations and maximum warmup time
  constexpr static int64_t minWarmupIter = 2;

  // maximum number of samples to take per trial
  constexpr static int64_t minSamples = 7;    // larger -> more accurate iid test
  constexpr static int64_t maxSamples = 500; // smaller -> faster iid test
  constexpr static int64_t maxTrials = 5; // how many times to try getting to iid
  constexpr static double maxTrialTime = 0.1; // maximum time allowed per trial

public:
  virtual ~Benchmark() {}
  struct Result {
    int64_t nTrials; // number of trial runs
    int64_t nIters;  // number of iterations in the final run
    double trimean;
    bool iid;
  };

  struct IterResult {
    double time;
  };

  // called first in run
  virtual void setup() {}
  virtual void teardown() {}

  // should be overridden to take a single sample and return it
  virtual IterResult run_iter() = 0;

  virtual Result run();
};

class MpiBenchmark : public Benchmark {
protected:
  MPI_Comm comm_;

public:
  MpiBenchmark(MPI_Comm comm) : comm_(comm) {}

  virtual Result run() override;
};
