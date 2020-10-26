#include "env.hpp"
#include "symbols.hpp"
#include "topology.hpp"

#include <metis.h>

extern "C" int
MPI_Dist_graph_create_adjacent(PARAMS_MPI_Dist_graph_create_adjacent) {
  if (environment::noTempi) {
    return libmpi.MPI_Dist_graph_create_adjacent(
        ARGS_MPI_Dist_graph_create_adjacent);
  }

  // call the underlying impl to actually create a communicator
  int err = libmpi.MPI_Dist_graph_create_adjacent(
      ARGS_MPI_Dist_graph_create_adjacent);

  // get the topology for the new communicator
  topology::cache_communicator(*comm_dist_graph);

  // partition the ranks by provided weights
  if (reorder) {

    idx_t *nvtxs;   // number of vertices
    idx_t ncon = 1; // number of balancing constraints (at least 1) (?)
    idx_t *xadj;    // adjacency structure (row pointers)
    idx_t *adjncy;  // adjacency structure (col indices)
    idx_t
        *vwgt; // weights of vertices. null for us to make all vertices the same

    idx_t *vsize;  // size of vertices. null for us to minimize edge cut
    idx_t *adjwgt; // weights of edges
    idx_t *nparts; // number of partitions for the graph
    real_t *tpwgts = nullptr; // equally divided among partitions
    real_t *ubvec = nullptr;  // load imbalance tolerance constraint is 1.001

    idx_t *options;
    idx_t *objval;
    idx_t *part; // stores the partition of each vertex

    int err =
        METIS_PartGraphKway(nvtxs, &ncon, xadj, adjncy, vwgt, vsize, adjwgt,
                            nparts, tpwgts, ubvec, options, objval, part);
    bool success = false;

    switch (err) {
    case METIS_OK: {
      success = true;
    }
    case METIS_ERROR_INPUT:
      LOG_ERROR("metis input error");
    case METIS_ERROR_MEMORY:
      LOG_ERROR("metis memory error");
    case METIS_ERROR:
      LOG_ERROR("metis other error");
    case default:
      LOG_ERROR("unhandled metis error");
    }

    if (!success) {
      LOG_ERROR("unable to reorder nodes");
    }
  }

  return err;
}
