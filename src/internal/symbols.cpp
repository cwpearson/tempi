//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "symbols.hpp"

#include "logging.hpp"

#include <dlfcn.h>
#include <link.h>

/* extern */ MpiFunc libmpi;

/* For reasons unknown, mvapich on Lassen is preloaded with OMPI_LD_PRELOAD_PREPEND
   this might cause dlsym(RTLD_NEXT) to fail to find various implementations
   when libtempi is predended to that variable
   so, if dlsym(RTLD_NEXT) fails, we explicitly look up where libmpi is and use that handle instead
*/


#define DLSYM2(A)                                                               \
  {                                                                            \
    libmpi.A = nullptr;                                                        \
    dlerror();                                                                 \
    libmpi.A = reinterpret_cast<Func_##A>(dlsym(RTLD_NEXT, #A));               \
    char *err = dlerror();                                                     \
    if (nullptr != err) {                                                      \
      LOG_FATAL("unabled to load " << #A << ": " << err);                      \
    } else {                                                                   \
      LOG_SPEW(#A << " at " << libmpi.A);                                      \
    }                                                                          \
  }

#define DLSYM(A)                                                               \
  {                                                                            \
    libmpi.A = nullptr;                                                        \
    dlerror();                                                                 \
    libmpi.A = reinterpret_cast<Func_##A>(dlsym(RTLD_NEXT, #A));               \
    char *err = dlerror();                                                     \
    if (nullptr != err) {                                                      \
      LOG_FATAL("unabled to load " << #A << " with RTLD_NEXT " << err);        \
    } \
    if (uintptr_t(libmpi.A) < uintptr_t(2)) { \
      LOG_SPEW(#A << " from RTLD_NEXT was NULL, using previously found libmpi.so"); \
      libmpi.A = reinterpret_cast<Func_##A>(dlsym(libmpiHandle, #A)); \
    }  \
    LOG_SPEW(#A << " at " << uintptr_t(libmpi.A)); \
  }

void init_symbols() {

  void *libmpiHandle = dlopen("libmpi.so", RTLD_LAZY);
  char origin[512];
  link_map *linkMap;
  dlinfo(libmpiHandle, RTLD_DI_ORIGIN, origin);
  dlinfo(libmpiHandle, RTLD_DI_LINKMAP, &linkMap);
  LOG_INFO("found libmpi.so at " << linkMap->l_name);

  DLSYM(MPI_Allgather);
  DLSYM(MPI_Alltoallv);
  DLSYM(MPI_Comm_free);
  DLSYM(MPI_Comm_rank);
  DLSYM(MPI_Comm_size);
  DLSYM(MPI_Comm_split_type);
  DLSYM(MPI_Dist_graph_create);
  DLSYM(MPI_Dist_graph_create_adjacent);
  DLSYM(MPI_Dist_graph_neighbors);
  DLSYM(MPI_Finalize);
  DLSYM(MPI_Get_library_version);
  DLSYM(MPI_Init);
  DLSYM(MPI_Init_thread);
  DLSYM(MPI_Irecv);
  DLSYM(MPI_Isend);
  DLSYM(MPI_Neighbor_alltoallv);
  DLSYM(MPI_Neighbor_alltoallw);
  DLSYM(MPI_Pack);
  DLSYM(MPI_Recv);
  DLSYM(MPI_Send);
  DLSYM(MPI_Send_init);
  DLSYM(MPI_Start);
  DLSYM(MPI_Test);
  DLSYM(MPI_Type_commit);
  DLSYM(MPI_Type_free);
  DLSYM(MPI_Unpack);
  DLSYM(MPI_Wait);
  DLSYM(MPI_Waitall);

  dlclose(libmpiHandle);
}
