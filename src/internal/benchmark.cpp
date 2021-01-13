//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "benchmark.hpp"
#include "iid.hpp"
#include "statistics.hpp"

#include <cassert>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::time_point<Clock, Duration> Time;

void Benchmark::estimate_nreps() {
  nreps_ = 1;
  for (int i = 0; i < 4; ++i) {
    Benchmark::Sample s = run_iter(); // measure time
    nreps_ = 200e-6 / s.time;         // measure for 200us
    nreps_ = std::max(nreps_, 1);
  }
}

void MpiBenchmark::estimate_nreps() {
  Benchmark::estimate_nreps();
  int rank;
  MPI_Comm_rank(comm_, &rank);
  MPI_Bcast(&nreps_, 1, MPI_INT, 0, comm_);
  if (0 == rank) {
    LOG_DEBUG("estimate nreps_=" << nreps_ << " for 200us");
  }
}

Benchmark::Result Benchmark::run(const RunConfig &rc) {

  // initialize benchmark resources
  setup();

  // warmup / estimate number of reps
  estimate_nreps();

  int64_t trial = 0;

  while (true) {
    int64_t iter = 0;
    Statistics stats;
    Time trialStart = Clock::now();
    while (iter < rc.minSamples ||
           (Duration(Clock::now() - trialStart).count() < rc.maxTrialSecs &&
            iter < rc.maxSamples)) {
      Sample res = run_iter();
      ++iter;
      stats.insert(res.time);
    }
    ++trial;

    if (sp_800_90B(stats.raw())) {
      return Result{.nTrials = trial,
                    .nIters = iter,
                    .trimean = stats.trimean(),
                    .iid = true};
    }
    if (trial == rc.maxTrials) {
      LOG_ERROR("benchmark ended without IID");
      return Result{.nTrials = trial,
                    .nIters = iter,
                    .trimean = stats.trimean(),
                    .iid = false};
    }
  }

  teardown();
}

Benchmark::Result MpiBenchmark::run(const RunConfig &rc) {
  assert(comm_);

  int rank, size;
  MPI_Comm_rank(comm_, &rank);
  MPI_Comm_size(comm_, &size);

  setup();
  estimate_nreps();
  assert(nreps_ > 0);

  int64_t trial = 0;
  while (true) {

    int64_t iter = 0;
    Statistics stats;
    const Time trialStart = Clock::now();
    bool runIter = true;
    while (runIter) {
      Sample res = run_iter();
      ++iter;
      Duration trialDur = Clock::now() - trialStart;
      stats.insert(res.time);
      runIter = iter < rc.minSamples ||
                (trialDur.count() < rc.maxTrialSecs && iter < rc.maxSamples);
      MPI_Bcast(&runIter, 1, MPI_C_BOOL, 0, comm_);
    }
    ++trial;

#if 0
      if (0 == rank) {
        for (size_t i = 0; i < 10; ++i) {
          std::cerr << stats.raw()[i] << " ";
        }
        std::cerr << "\n";
      }
      MPI_Barrier(comm_);
#endif

#if 0
      std::cerr << rank << ":: "
                << "trial: " << trial << " iter: " << iter
                << " trimean: " << stats.trimean() << "\n";
      MPI_Barrier(comm_);
#endif

    bool runNextTrial = true;
    bool iid = false;
    if (0 == rank && sp_800_90B(stats.raw())) {
      runNextTrial = false;
      iid = true;
    }
    if (0 == rank && trial == rc.maxTrials) {
      LOG_ERROR("benchmark ended without IID");
      runNextTrial = false;
      iid = false;
    }
    MPI_Bcast(&runNextTrial, 1, MPI_C_BOOL, 0, comm_);
    MPI_Bcast(&iid, 1, MPI_C_BOOL, 0, comm_);
    if (!runNextTrial) {
      return Result{.nTrials = trial,
                    .nIters = iter,
                    .trimean = stats.trimean(),
                    .iid = iid};
    }
  }

  teardown();
}
