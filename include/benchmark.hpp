//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <cstdint>

#include <mpi.h>

class Benchmark {

protected:
  int nreps_;

  // run a few samples to estimate the number of reps per sample
  virtual void estimate_nreps();

public:
  Benchmark() : nreps_(1) {}
  virtual ~Benchmark();
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
    int64_t minSamples;  // at least this many benchmark samples per trial
    int64_t maxSamples;  // no more than this many benchmark samples per trial
    int64_t maxTrials;   // this man attempts to get iid
    double maxTrialSecs; // no more than this much wall time per trial

    // default configuration
    RunConfig() {
      minSamples = 7;
      maxSamples = 500;
      maxTrials = 10;
      maxTrialSecs = 1;
    }
  };

  // before first sample
  virtual void setup();

  // after last sample
  virtual void teardown();
  virtual Sample run_iter() = 0; // average of n operations

  virtual Result run(const RunConfig &rc);
};

class MpiBenchmark : public Benchmark {
protected:
  MPI_Comm comm_;

  virtual void estimate_nreps() override;

public:
  MpiBenchmark(MPI_Comm comm) : Benchmark(), comm_(comm) {}

  virtual Result run(const RunConfig &rc) override;
};
