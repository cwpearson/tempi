//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "allocators.hpp"
#include "async_operation.hpp"
#include "counters.hpp"
#include "env.hpp"
#include "events.hpp"
#include "logging.hpp"
#include "streams.hpp"
#include "worker.hpp"

#include <mpi.h>

#include <dlfcn.h>

#define ARGS // empty args

extern "C" int MPI_Finalize() {
  typedef int (*Func_MPI_Finalize)();
  static Func_MPI_Finalize fn = nullptr;
  if (!fn) {
    fn = reinterpret_cast<Func_MPI_Finalize>(dlsym(RTLD_NEXT, "MPI_Finalize"));
  }
  // TODO: in tests, it's possible that TEMPI was enabled during MPI_Init and
  // disabled now
  TEMPI_DISABLE_GUARD;

  async::finalize();
  events::finalize();
  worker_finalize();
  streams_finalize();
  allocators::finalize();
  counters::finalize();

  LOG_SPEW("library MPI_Finalize");
  int err = fn();
  return err;
}
