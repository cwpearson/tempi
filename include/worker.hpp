#pragma once

#include <mpi.h>

#include <thread>

extern std::thread workerThread;

void worker_init();
void worker_finalize();

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
