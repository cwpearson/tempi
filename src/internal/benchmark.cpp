#include "benchmark.hpp"
#include "iid.hpp"
#include "statistics.hpp"

#include <cassert>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::time_point<Clock, Duration> Time;

Benchmark::Result Benchmark::run(const RunConfig &rc) {

  setup();

  for (int iter = 0; iter < rc.minWarmupSamples; ++iter) {
    run_iter();
  }

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

  {
    int iter = 0;
    bool keepRunning = true;
    while (keepRunning) {
      run_iter();
      ++iter;
      keepRunning = iter < rc.minWarmupSamples;
      MPI_Bcast(&keepRunning, 1, MPI_C_BOOL, 0, comm_);
    }
    MPI_Barrier(comm_);

#if 0
    std::cerr << rank << ":: "
              << "warmup:"
              << " iter: " << iter << "\n";
#endif
  }

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
