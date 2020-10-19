#include "symbols.hpp"

#include <dlfcn.h>

/* extern */ MpiFunc libmpi;

#define DLSYM(A) libmpi.A = reinterpret_cast<Func_##A>(dlsym(RTLD_NEXT, #A))

void init_symbols() {
  DLSYM(MPI_Init);
  DLSYM(MPI_Init_thread);
  DLSYM(MPI_Isend);
  DLSYM(MPI_Get_library_version);
}