//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <mpi.h>

#include <thread>

extern std::thread workerThread;

/* holds the specification for a job delivered to the worker
 */
struct WorkerJob {
  //   enum class Kind { ISEND };

  struct IsendParams {
    const void *buf;
    int count;
    MPI_Datatype datatype;
    int dest;
    int tag;
    MPI_Comm comm;
    MPI_Request *request;
  };

  enum class Kind { ISEND };
  constexpr static Kind ISEND = Kind::ISEND;

  Kind kind;
  union {
    IsendParams isend;
  } params;
};

/* holds the state for a persistent task the worker is executing.

  should be derived from for different tasks
 */
class WorkerTask {
public:
  virtual ~WorkerTask() {}

  // ready to call `progress`
  virtual bool ready() = 0;
  virtual void progress() = 0;
  // true if task is complete
  virtual bool done() = 0;
};

void worker_init();
void worker_finalize();
void worker_push(const WorkerJob &job);