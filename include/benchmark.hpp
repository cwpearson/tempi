#pragma once

#include <cstdint>

#include <mpi.h>

class Benchmark {

public:
  virtual ~Benchmark() {}
  struct Result {
    int64_t nTrials; // number of trial runs
    int64_t nIters;  // number of iterations in the final run
    double trimean;
    bool iid;
  };

  struct Sample {
    double time;
  };

  struct RunConfig {
    int64_t minWarmupSamples;  // at least this many iterations during warmup
    int64_t minSamples;        // at least this many benchmark samples
    int64_t maxSamples;  // no more than this many benchmark samples
    int64_t maxTrials;     // this man attempts to get iid
    double maxTrialSecs; // no more than this much wall time per trial

    // default configuration
    RunConfig() {
      minWarmupSamples = 2;
      minSamples = 7;
      maxSamples = 500;
      maxTrials = 10;
      maxTrialSecs = 1;
    }
  };

  virtual void setup() {}        // before first sample
  virtual void teardown() {}     // after last sample
  virtual Sample run_iter() = 0; // average of n operations

  virtual Result run(const RunConfig &rc);
};

class MpiBenchmark : public Benchmark {
protected:
  MPI_Comm comm_;

public:
  MpiBenchmark(MPI_Comm comm) : comm_(comm) {}

  virtual Result run(const RunConfig &rc) override;
};
