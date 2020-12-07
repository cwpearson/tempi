#include "benchmark.hpp"
#include "iid.hpp"
#include "statistics.hpp"

#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::time_point<Clock, Duration> Time;

Benchmark::Result Benchmark::run() {

  int64_t trial = 0;

  while (true) {

    // warmup for 5 iters or 0.01s, whichever comes first
    {
      Time start = Clock::now();
      Duration warmupSecs{};
      int iter = 0;
      while (warmupSecs.count() < maxWarmupSecs && iter < maxWarmupIter) {
        run_iter();
        ++iter;
        warmupSecs = Clock::now() - start;
      }
    }

    // run iterations for at least minIters and at least maxSecsUnderTest
    {
      int64_t iter = 0;
      Statistics stats;
      Time wholeStart = Clock::now();
      Duration wholeTime{};
      while (iter < minIters || wholeTime.count() < maxTrialTime) {
        IterResult res = run_iter();
        ++iter;
        wholeTime = Clock::now() - wholeStart;
        stats.insert(res.time);
      }
      ++trial;

      if (sp_800_90B(stats.raw())) {
        return Result{
            .nTrials = trial, .nIters = iter, .trimean = stats.trimean()};
      }
      if (trial == maxTrials) {
        LOG_ERROR("benchmark ended without IID");
        return Result{
            .nTrials = trial, .nIters = iter, .trimean = stats.trimean()};
      }
    }
  }
}