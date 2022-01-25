//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "allocators.hpp"
#include "counters.hpp"
#include "env.hpp"
#include "events.hpp"
#include "logging.hpp"
#include "measure_system.hpp"
#include "streams.hpp"
#include "symbols.hpp"
#include "topology.hpp"
#include "types.hpp"
#include "tags.hpp"

#include <mpi.h>

#include <nvToolsExt.h>

extern "C" int MPI_Init_thread(PARAMS_MPI_Init_thread) {

  // before anything else, read env vars to control tempi
  init_symbols();
  read_environment();
  if (environment::noTempi) {
    return libmpi.MPI_Init_thread(ARGS_MPI_Init_thread);
  }

  LOG_INFO("in TEMPI's MPI_Init_thread!");

  LOG_SPEW("call " << libmpi.MPI_Init_thread);
  int err = libmpi.MPI_Init_thread(ARGS_MPI_Init_thread);
  // can use logging now that MPI_Init has been called
  LOG_SPEW("finished library MPI_Init_thread");

  if (MPI_THREAD_SINGLE == *provided) {
    LOG_SPEW("MPI_THREAD_SINGLE");
  } else if (MPI_THREAD_FUNNELED == *provided) {
    LOG_SPEW("MPI_THREAD_FUNNELED");
  } else if (MPI_THREAD_SERIALIZED == *provided) {
    LOG_SPEW("MPI_THREAD_SERIALIZED");
  } else if (MPI_THREAD_MULTIPLE == *provided) {
    LOG_SPEW("MPI_THREAD_MULTIPLE");
  }

  int rank;
  libmpi.MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (0 == rank) {
    LOG_INFO("MPI_Wtick() = " << MPI_Wtick());
  }

  tags::init();
  counters::init();
  allocators::init();
  events::init();
  topology_init();
  streams_init();
  types_init();
  tempi::system::init();

  return err;
}
