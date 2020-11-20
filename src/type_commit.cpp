#include "env.hpp"
#include "logging.hpp"
#include "packer.hpp"
#include "symbols.hpp"
#include "types.hpp"

#include <mpi.h>

extern "C" int MPI_Type_commit(PARAMS_MPI_Type_commit) {
  int result = libmpi.MPI_Type_commit(ARGS_MPI_Type_commit);

  if (environment::noTempi) {
    return result;
  }
  if (environment::noTypeCommit) {
    return result;
  }

  if (MPI_SUCCESS != result) {
    LOG_ERROR("error in system MPI_Type_commit call");
    return result;
  }

  Type type = traverse(*datatype);
  if (packerCache.count(*datatype)) {
    LOG_SPEW("found MPI_Datatype " << uintptr_t(*datatype)
                                   << " in packerCache");
    return result;
  } else {
    std::unique_ptr<Packer> pPacker = plan_pack(type);
    if (pPacker) {
      packerCache[*datatype] = std::move(pPacker);
      LOG_SPEW("cache packer for datatype=" << uintptr_t(*datatype));
    } else {
      LOG_WARN("packer for " << uintptr_t(*datatype) << " was null");
    }
  }

  return result;
}