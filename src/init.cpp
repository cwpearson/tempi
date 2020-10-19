#include "env.hpp"
#include "logging.hpp"
#include "streams.hpp"
#include "symbols.hpp"
#include "topology.hpp"
#include "worker.hpp"

#include <mpi.h>

#include <nvToolsExt.h>

extern "C" int MPI_Init(PARAMS_MPI_Init) {

  // before anything else, read env vars to control tempi
  init_symbols();
  read_environment();
  static Func_MPI_Init fn = libmpi.MPI_Init;
  if (environment::noTempi) {
    return fn(ARGS_MPI_Init);
  }

  int err = fn(ARGS_MPI_Init);
  // can use logging now that MPI_Init has been called
  LOG_DEBUG("finished library MPI_Init");

  int provided;
  MPI_Query_thread(&provided);
  if (MPI_THREAD_SINGLE == provided) {
    LOG_DEBUG("MPI_THREAD_SINGLE");
  } else if (MPI_THREAD_FUNNELED) {
    LOG_DEBUG("MPI_THREAD_FUNNELED");
  } else if (MPI_THREAD_SERIALIZED) {
    LOG_DEBUG("MPI_THREAD_SERIALIZED");
  } else if (MPI_THREAD_MULTIPLE) {
    LOG_DEBUG("MPI_THREAD_MULTIPLE");
  }

  topology_init();
  streams_init();
  worker_init();

  return err;
}
