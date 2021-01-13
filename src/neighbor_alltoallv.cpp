//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "cuda_runtime.hpp"
#include "env.hpp"
#include "logging.hpp"
#include "streams.hpp"
#include "symbols.hpp"
#include "topology.hpp"
#include "types.hpp"

#include <string>

extern "C" int MPI_Neighbor_alltoallv(PARAMS_MPI_Neighbor_alltoallv) {
  if (environment::noTempi) {
    return libmpi.MPI_Neighbor_alltoallv(ARGS_MPI_Alltoallv);
  }

#if TEMPI_OUTPUT_LEVEL >= 4
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
#endif

  /* wait for any GPU packing operation to finish if all are true
   1) packed datatype
   2) GPU buffer

   we can't tell what the original datatype was, so we can't check if we
   launched a GPU packer
  */

  if (MPI_PACKED == sendtype) {
    cudaPointerAttributes attr{};
    CUDA_RUNTIME(cudaPointerGetAttributes(&attr, sendbuf));
    if (nullptr != attr.devicePointer) {
      LOG_SPEW("Neighbor_alltoallv: sync packer");
      CUDA_RUNTIME(cudaStreamSynchronize(kernStream));
    }
  }

  return libmpi.MPI_Neighbor_alltoallv(ARGS_MPI_Alltoallv);
}
