#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "symbols.hpp"
#include "topology.hpp"
#include "types.hpp"
#include "streams.hpp"

#include <string>

extern "C" int MPI_Neighbor_alltoallv(PARAMS_MPI_Neighbor_alltoallv) {
  if (environment::noTempi) {
    return libmpi.MPI_Neighbor_alltoallv(ARGS_MPI_Alltoallv);
  }

  {
    auto it = degrees.find(comm);
    if (it != degrees.end()) {

      /* this call does not take ranks, so there is no need to handle
         reordering. The library ranks are different than the application ranks,
         but they have the right neighbors in a consistent order, just with
         different numbers
      */

      {
        std::string s;
        for (int i = 0; i < it->second.indegree; ++i) {
          s += std::to_string(sendcounts[i]) + " ";
        }
        LOG_SPEW("sendcounts=" << s);
      }

      {
        std::string s;
        for (int i = 0; i < it->second.outdegree; ++i) {
          s += std::to_string(recvcounts[i]) + " ";
        }
        LOG_SPEW("recvcounts=" << s);
      }
    }
  }

  /* wait for any GPU packing operation to finish if all are true
   1) packed datatype
   2) there is a packer for the type
   3) GPU buffer
  */

  if (MPI_PACKED == sendtype) {
    auto it = packerCache.find(sendtype);
    if (packerCache.end() != it) {
      cudaPointerAttributes attr{};
      CUDA_RUNTIME(cudaPointerGetAttributes(&attr, sendbuf));
      if (nullptr != attr.devicePointer) {
        LOG_SPEW("Neighbor_alltoallv: sync packer");
        CUDA_RUNTIME(cudaStreamSynchronize(kernStream[attr.device]));
      }
    }
  }

  return libmpi.MPI_Neighbor_alltoallv(ARGS_MPI_Alltoallv);
}
