#include "env.hpp"
#include "logging.hpp"
#include "packer.hpp"
#include "types.hpp"

#include <cuda_runtime.h>
#include <mpi.h>

#include <dlfcn.h>

#include <cassert>

#define PARAMS MPI_Datatype *datatype

#define ARGS datatype

extern "C" int MPI_Type_commit(PARAMS) {
  LOG_DEBUG("MPI_Type_commit");

  // find the underlying MPI call
  typedef int (*Func_MPI_Type_commit)(PARAMS);
  static Func_MPI_Type_commit fn = nullptr;
  if (!fn) {
    fn = reinterpret_cast<Func_MPI_Type_commit>(
        dlsym(RTLD_NEXT, "MPI_Type_commit"));
  }

  int result = fn(ARGS);

  bool enabled = true;
  enabled &= (!environment::noTypeCommit);

  if (enabled) {
    Type type = traverse(*datatype);
    if (packerCache.count(*datatype)) {
      return result;
    } else {
      std::shared_ptr<Packer> pPacker = plan_pack(type);
      if (pPacker) {
        packerCache[*datatype] = plan_pack(type);
      }
    }
  }

  if (MPI_SUCCESS != result) {
    LOG_ERROR("error in underlying MPI call");
    return result;
  }
  return result;
}