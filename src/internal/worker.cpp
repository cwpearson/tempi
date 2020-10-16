#include "worker.hpp"

#include "logging.hpp"
#include "queue.hpp"
#include "task_staged_isend.hpp"

#include <nvToolsExt.h>
// #include <sys/types.h> // gettid
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <vector>

struct WorkerParams {
  std::atomic<bool> &shutdown;
  Queue<WorkerJob> &queue;
  std::vector<WorkerTask *> &pending;
};

WorkerTask *make_task(const WorkerJob &job) {
  switch (job.kind) {
  case WorkerJob::ISEND: {
    return new StagedIsend(job.params.isend.buf, job.params.isend.count,
                           job.params.isend.datatype, job.params.isend.dest,
                           job.params.isend.tag, job.params.isend.comm,
                           job.params.isend.request);
  }
  default:
    LOG_FATAL("no task defined for job");
  }
}

void worker_loop(WorkerParams params) {

  typedef std::chrono::high_resolution_clock Clock;
  typedef std::chrono::duration<double> Duration;
  typedef std::chrono::time_point<Clock> Time;

  nvtxNameOsThread(gettid(), "worker");

  while (true) {
    nvtxMark("workerThread loop");

    Time start = Clock::now();

    // check for shutdown
    if (params.shutdown) {
      LOG_DEBUG("worker saw shutdown request");
      return;
    }

    // do all new work that has been assigned to us
    nvtxRangePush("queue loop");
    while (!params.queue.empty()) {
      LOG_SPEW("workerThread pop job");
      WorkerJob job = params.queue.pop();
      params.pending.push_back(make_task(job));
    }
    nvtxRangePop();

    // check status of any existing work
    nvtxRangePush("task loop");
  task_loop:
    // LOG_SPEW(params.pending.size() << " pending workerThread tasks");
    for (size_t i = 0; i < params.pending.size(); ++i) {
      WorkerTask *task = params.pending[i];
      // if done, erase and restart iteration
      if (task->done()) {
        params.pending.erase(params.pending.begin() + i);
        delete task;
        goto task_loop;
      } else if (task->ready()) { // if ready, progress and restart iteration
        LOG_SPEW("progressed task");
        task->progress();
        goto task_loop;
      }
    }
    nvtxRangePop();

    /* check for new work to  do every 10us.
    TODO:
    ideally, we would sleep until
    1) a job was put in the queue
    2) a task needed some work
    3) we are supposed to shut down
    May be able to use a single condition variable that is notified when any of
    those things happens.
    */
    Duration sleepTime = Duration(0.00005) - Duration(Clock::now() - start);
    std::cerr << sleepTime.count() << "\n";
    if (sleepTime > Duration(0)) {
      nvtxRangePush("sleep");
      std::this_thread::sleep_for(sleepTime);
      nvtxRangePop();
    }
  }
}

/*extern*/ std::thread workerThread;

std::atomic<bool> workerShutdown;
Queue<WorkerJob> workerQueue;
std::vector<WorkerTask *> workerPending;

void worker_init() {
  #if 0
  nvtxRangePush("worker_init");
  workerShutdown = false;
  WorkerParams params{.shutdown = workerShutdown,
                      .queue = workerQueue,
                      .pending = workerPending};

  std::thread worker(worker_loop, params);
  LOG_DEBUG("started workerThread");
  workerThread = std::move(worker);
  nvtxRangePush("sleep");
  std::this_thread::sleep_for(std::chrono::microseconds(200));
  nvtxRangePop();
  nvtxRangePop();
  #endif
}

void worker_finalize() {
  #if 0
  nvtxRangePush("worker_finalize");
  workerShutdown = true;
  LOG_DEBUG("sent shutdown to workerThread. waiting...");
  workerThread.join();
  LOG_DEBUG("workerThread joined");
  nvtxRangePop();
  #endif
}

void worker_push(const WorkerJob &job) {
  LOG_DEBUG("submitting job");
  workerQueue.push(job);
}