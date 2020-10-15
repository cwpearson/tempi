#include "worker.hpp"

#include "logging.hpp"
#include "queue.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <vector>

struct WorkerParams {
  std::atomic<bool> &shutdown;
  Queue<WorkerJob> &queue;
  std::vector<WorkerTask *> &pending;
};

void worker_loop(WorkerParams params) {

  typedef std::chrono::system_clock Clock;
  typedef std::chrono::duration<double> Duration;
  typedef std::chrono::time_point<Clock> Time;

  while (true) {

    Time start = Clock::now();

    // check for shutdown
    if (params.shutdown) {
      LOG_DEBUG("worker saw shutdown request");
      return;
    }

    // do all new work that has been assigned to us
    while (!params.queue.empty()) {
      LOG_SPEW("workerThread pop job");
      WorkerJob job = params.queue.pop();
    }

  // check status of any existing work
  task_loop:
    for (size_t i = 0; i < params.pending.size(); ++i) {
      WorkerTask *task = params.pending[i];
      // if done, erase and restart iteration
      if (task->done()) {
        params.pending.erase(params.pending.begin() + i);
        delete task;
        goto task_loop;
      } else if (task->ready()) { // if ready, progress and restart iteration
        task->progress();
        goto task_loop;
      }
    }

    /* check for new work to  do every 5us.
    TODO:
    ideally, we would sleep until
    1) a job was put in the queue
    2) a task needed some work
    3) we are supposed to shut down
    May be able to use a single condition variable that is notified when any of
    those things happens.
    */
    Duration sleepTime = Duration(0.000005) - Duration(Clock::now() - start);
    if (sleepTime > Duration(0)) {
      std::this_thread::sleep_for(sleepTime);
    }
  }
}

/*extern*/ std::thread workerThread;

std::atomic<bool> workerShutdown;
Queue<WorkerJob> workerQueue;
std::vector<WorkerTask *> workerPending;

void worker_init() {
  workerShutdown = false;
  WorkerParams params{.shutdown = workerShutdown,
                      .queue = workerQueue,
                      .pending = workerPending};

  std::thread worker(worker_loop, params);
  LOG_DEBUG("started workerThread");
  workerThread = std::move(worker);
}

void worker_finalize() {
  workerShutdown = true;
  LOG_DEBUG("sent shutdown to workerThread. waiting...");
  workerThread.join();
  LOG_DEBUG("workerThread joined");
}