#include "logging.hpp"

#include "type_commit.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <dlfcn.h>

#include <cstdlib>
#include <vector>

extern "C" int MPI_Pack(const void *inbuf, int incount, MPI_Datatype datatype,
                        void *outbuf, int outsize, int *position,
                        MPI_Comm comm) {
  LOG_DEBUG("MPI_Pack");

  // find the underlying MPI call
  typedef int (*Func_MPI_Pack)(const void *inbuf, int incount,
                               MPI_Datatype datatype, void *outbuf, int outsize,
                               int *position, MPI_Comm comm);
  static Func_MPI_Pack fn = nullptr;
  if (!fn) {
    fn = reinterpret_cast<Func_MPI_Pack>(dlsym(RTLD_NEXT, "MPI_Pack"));
  }

  bool enabled = true;
  enabled &= (nullptr == std::getenv("SCAMPI_NO_PACK"));

  if (enabled && packerCache.count(datatype)) {
    std::shared_ptr<Packer> packer = packerCache[datatype];
    packer->pack(outbuf, position, inbuf, incount);
    return MPI_SUCCESS;
  } else {
    LOG_DEBUG("defer to MPI_Pack");
    return fn(inbuf, incount, datatype, outbuf, outsize, position, comm);
  }
}