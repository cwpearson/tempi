#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "requests.hpp"

#include <mpi.h>

#include <dlfcn.h>

#define PARAMS MPI_Request *request, MPI_Status *status
#define ARGS request, status

extern "C" int MPI_Wait(PARAMS) {
  typedef int (*Func_MPI_Wait)(PARAMS);
  static Func_MPI_Wait fn = nullptr;
  if (!fn) {
    fn = reinterpret_cast<Func_MPI_Wait>(dlsym(RTLD_NEXT, "MPI_Wait"));
  }
  TEMPI_DISABLE_GUARD;
  LOG_DEBUG("MPI_Wait");

  wait(request);

  return fn(ARGS);

#if 0
  // FIXME:
  // we need to make sure library MPI_Wait is called before library MPI_Wait.
  // it may be a while before library MPI_Wait is called, even though we return
  // immediately and app may call MPI_Wait may be called right away. so, we need
  // to track that this request should not be passed onto the library MPI_Wait
  // yet. and MPI_Wait needs to make sure
#endif
}
