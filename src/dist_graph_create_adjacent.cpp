#include "env.hpp"
#include "logging.hpp"
#include "partition.hpp"
#include "symbols.hpp"
#include "topology.hpp"

#include <metis.h>
#include <nvToolsExt.h>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <tuple>

extern "C" int
MPI_Dist_graph_create_adjacent(PARAMS_MPI_Dist_graph_create_adjacent) {
  if (environment::noTempi) {
    return libmpi.MPI_Dist_graph_create_adjacent(
        ARGS_MPI_Dist_graph_create_adjacent);
  }

  /* Two options:
     1)
     pass arguments directly to the library call, to create a graph communicator
     Based on which ranks the resulting nodes end up on, figure out which
     library rank i should provide application rank j.

     The problem here is that rank j's edges are not the same number as i's, so
     we can't directly use neighborhood calls, and have to reimplement all of
     them,

     2) figure out which old communicator rank i should run which new
     communicator rank j then have rank i provide j's edges to the call with
     reorder set to 0

     Then, in the child communicator
     MPI_Comm_rank(i) -> j
     MPI_Dist_neighbors will return translation from i -> j
     MPI_Neighbor_alltoallv will transparently work
  */

  const size_t numNodes = topology::num_nodes(comm_old);

  int oldSize, oldRank;
  libmpi.MPI_Comm_rank(comm_old, &oldRank);
  libmpi.MPI_Comm_size(comm_old, &oldSize);

  if (numNodes > 1) {

    // the partition the rank should belong to
    std::vector<int> part(oldSize, -1);

    // random placement
    if (reorder && PlacementMethod::RANDOM == environment::placement) {

      // assign each rank to a random partition
      part = partition::random(oldSize, numNodes);

    } else if (reorder && PlacementMethod::METIS == environment::placement) {

      // build edgelist on every node
      std::vector<int> edgeSrc, edgeDst, weight;
      for (int i = 0; i < indegree; ++i) {
        edgeSrc.push_back(sources[i]);
        edgeDst.push_back(oldRank);
        weight.push_back(sourceweights[i]);
      }
      for (int i = 0; i < outdegree; ++i) {
        edgeSrc.push_back(oldRank);
        edgeDst.push_back(destinations[i]);
        weight.push_back(destweights[i]);
      }

      // get edge counts from all ranks
      int edgeCnt = indegree + outdegree;
      std::vector<int> edgeCnts(oldSize);
      MPI_Gather(&edgeCnt, 1, MPI_INT, edgeCnts.data(), 1, MPI_INT, 0,
                 comm_old);

      std::vector<int> edgeOffs(edgeCnts.size(), 0);
      if (0 == oldRank) {
        // recieve edges from all ranks
        for (int i = 1; i < oldSize; ++i) {
          edgeOffs[i] = edgeOffs[i - 1] + edgeCnts[i - 1];
        }
        edgeSrc.resize(edgeOffs[oldSize - 1] + edgeCnts[oldSize - 1]);
        edgeDst.resize(edgeOffs[oldSize - 1] + edgeCnts[oldSize - 1]);
        weight.resize(edgeOffs[oldSize - 1] + edgeCnts[oldSize - 1]);
      }

      // get edge data from all ranks
      MPI_Gatherv(edgeSrc.data(), edgeCnt, MPI_INT, edgeSrc.data(),
                  edgeCnts.data(), edgeOffs.data(), MPI_INT, 0, comm_old);
      MPI_Gatherv(edgeDst.data(), edgeCnt, MPI_INT, edgeDst.data(),
                  edgeCnts.data(), edgeOffs.data(), MPI_INT, 0, comm_old);
      MPI_Gatherv(weight.data(), edgeCnt, MPI_INT, weight.data(),
                  edgeCnts.data(), edgeOffs.data(), MPI_INT, 0, comm_old);

      // build CSR on root node
      if (0 == oldRank) {
        std::vector<std::tuple<int, int, int>> edges; // src dst weight
        for (size_t i = 0; i < edgeSrc.size(); ++i) {
          edges.push_back(std::make_tuple(edgeSrc[i], edgeDst[i], weight[i]));
        }

        std::sort(edges.begin(), edges.end());

        // debug output
        {
          std::string s;
          for (auto &e : edges) {
            s += "[" + std::to_string(std::get<0>(e)) + "," +
                 std::to_string(std::get<1>(e)) + "," +
                 std::to_string(std::get<2>(e)) + "] ";
          }
          LOG_SPEW("edges: " << s);
        }

        // delete self edges
        for (size_t i = 0; i < edges.size(); ++i) {
          if (std::get<0>(edges[i]) == std::get<1>(edges[i])) {
            edges.erase(edges.begin() + i, edges.begin() + i + 1);
          }
        }

        // delete duplicate edges
        bool changed = true;
        while (changed) {
          changed = false;
          for (int64_t i = 0; i < int64_t(edges.size()) - 1; ++i) {
            auto lb =
                std::lower_bound(edges.begin() + i + 1, edges.end(), edges[i]);
            auto ub =
                std::upper_bound(edges.begin() + i + 1, edges.end(), edges[i]);
            if (lb != ub) {
              edges.erase(lb, ub);
              changed = true;
              break;
            }
          }
        }

        // bidirectional edge weights must be the same for METIS.
        // sum up the two directions

        // comparator ignoring weight
        auto ignore_weight = [](const std::tuple<int, int, int> &a,
                                const std::tuple<int, int, int> &b) {
          return std::make_pair(std::get<0>(a), std::get<1>(a)) <
                 std::make_pair(std::get<0>(b), std::get<1>(b));
        };
        for (int64_t i = 0; i < int64_t(edges.size()) - 1; ++i) {
          // back edge with no weight
          std::tuple<int, int, int> backedge(std::get<1>(edges[i]),
                                             std::get<0>(edges[i]), 0);

          // find the position of the back edge
          auto lb = std::lower_bound(edges.begin() + i + 1, edges.end(),
                                     backedge, ignore_weight);
          auto ub = std::upper_bound(edges.begin() + i + 1, edges.end(),
                                     backedge, ignore_weight);
          if (lb != ub) {
            std::get<2>(edges[i]) += std::get<2>(*lb);
            std::get<2>(*lb) = std::get<2>(edges[i]);
          }
        }

        // debug output
        {
          std::string s;
          for (auto &e : edges) {
            s += "[" + std::to_string(std::get<0>(e)) + "," +
                 std::to_string(std::get<1>(e)) + "," +
                 std::to_string(std::get<2>(e)) + "] ";
          }
          LOG_DEBUG("edges: " << s);
        }

        // build CSR
        std::vector<idx_t> xadj;   // adjacency structure (row pointers)
        std::vector<idx_t> adjncy; // adjacency structure (col indices)
        std::vector<idx_t> adjwgt; // weight of edges
        idx_t rp = 0;
        for (size_t ei = 0; ei < edges.size(); ++ei) {
          for (; rp <= std::get<0>(edges[ei]); ++rp) {
            xadj.push_back(ei);
          }
          adjncy.push_back(std::get<1>(edges[ei]));
          adjwgt.push_back(std::get<2>(edges[ei]));
        }
        for (; rp <= oldSize; ++rp) {
          xadj.push_back(adjncy.size());
        }
        assert(xadj.size() == size_t(oldSize) + 1);

        // debug output
        {
          std::string s;
          for (idx_t e : xadj) {
            s += std::to_string(e) + " ";
          }
          LOG_DEBUG("xadj: " << s);
        }
        {
          std::string s;
          for (idx_t e : adjncy) {
            s += std::to_string(e) + " ";
          }
          LOG_DEBUG("adjncy: " << s);
        }
        {
          std::string s;
          for (idx_t e : adjwgt) {
            s += std::to_string(e) + " ";
          }
          LOG_DEBUG("adjwgt: " << s);
        }

        idx_t nvtxs = oldSize; // number of vertices
        idx_t ncon = 1; // number of balancing constraints (at least 1) (?)
        idx_t *vwgt = nullptr; // weights of vertices. null for us to make all
                               // vertices the same
        idx_t *vsize =
            nullptr; // size of vertices. null for us to minimize edge cut
        idx_t nparts = numNodes;  // number of partitions for the graph
        real_t *tpwgts = nullptr; // equally divided among partitions
        real_t *ubvec = nullptr; // load imbalance tolerance constraint is 1.001
        idx_t options[METIS_NOPTIONS]{};

        // kway options. comment out means 0 default is okay
        METIS_SetDefaultOptions(options);
        // options[METIS_OPTION_DBGLVL] = 1;
        idx_t objval;

        nvtxRangePush("METIS_PartGraphKway");
        static_assert(sizeof(idx_t) == sizeof(int), "wrong metis idx_t");
        int metisErr =
            METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), vwgt,
                                vsize, adjwgt.data(), &nparts, tpwgts, ubvec,
                                options, &objval, part.data());
        nvtxRangePop();
        bool success = false;

        switch (metisErr) {
        case METIS_OK: {
          success = true;
          break;
        }
        case METIS_ERROR_INPUT:
          LOG_FATAL("metis input error");
          break;
        case METIS_ERROR_MEMORY:
          LOG_FATAL("metis memory error");
          break;
        case METIS_ERROR:
          LOG_FATAL("metis other error");
          break;
        default:
          LOG_FATAL("unexpected METIS error");
          break;
        }

        if (!success) {
          LOG_ERROR("unable to reorder nodes");
        }

        LOG_DEBUG("METIS objval=" << objval);
        {
          std::string s;
          for (idx_t i : part) {
            s += std::to_string(i) + " ";
          }
          LOG_DEBUG("part=" << s);
        }

      } // 0 == graphRank

      // broadcast the partition assignment to all nodes
      MPI_Bcast(part.data(), part.size(), MPI_INT, 0, comm_old);
    }

#if TEMPI_OUTPUT_LEVEL >= 4
    if (0 == oldRank) {
      std::string s("node assignment app rank: ");
      for (int r : part) {
        s += std::to_string(r) + " ";
      }
      LOG_SPEW(s);
    }
#endif

    /* library rank i (this rank) is presented as application rank j,
     rank j needs to send edge information to rank i for the graph creation
     call.
     The edge information needs to be passed through the transformation
     before it is provided to the underlying library as well.
     the sources and destinations need to be adjusted accordingly so the
     library ranks have the right neighbors
    */
    Placement placement = topology::make_placement(comm_old, part);

    // transform sources and destinations to the library rank that will run
    // them
    std::vector<int> txsources(indegree), txdestinations(outdegree);
    for (int i = 0; i < indegree; ++i) {
      txsources[i] = placement.libRank[sources[i]];
    }
    for (int i = 0; i < outdegree; ++i) {
      txdestinations[i] = placement.libRank[destinations[i]];
    }

    int libindegree, liboutdegree;

    // indegree and outdegree
    // this rank's indegree needs to be sent to the rank that will run it
    // this rank needs the indegree of the app rank that it runs
    MPI_Sendrecv(&indegree, 1, MPI_INT, placement.libRank[oldRank], 0,
                 &libindegree, 1, MPI_INT, placement.appRank[oldRank], 0,
                 comm_old, MPI_STATUS_IGNORE);
    // outdegree
    MPI_Sendrecv(&outdegree, 1, MPI_INT, placement.libRank[oldRank], 0,
                 &liboutdegree, 1, MPI_INT, placement.appRank[oldRank], 0,
                 comm_old, MPI_STATUS_IGNORE);

    std::vector<int> libsources(libindegree), libsourceweights(libindegree),
        libdestinations(liboutdegree), libdestweights(liboutdegree);
    // edge data
    MPI_Sendrecv(txsources.data(), indegree, MPI_INT,
                 placement.libRank[oldRank], 0, libsources.data(),
                 libsources.size(), MPI_INT, placement.appRank[oldRank], 0,
                 comm_old, MPI_STATUS_IGNORE);
    MPI_Sendrecv(sourceweights, indegree, MPI_INT, placement.libRank[oldRank],
                 0, libsourceweights.data(), libsourceweights.size(), MPI_INT,
                 placement.appRank[oldRank], 0, comm_old, MPI_STATUS_IGNORE);
    MPI_Sendrecv(txdestinations.data(), outdegree, MPI_INT,
                 placement.libRank[oldRank], 0, libdestinations.data(),
                 libdestinations.size(), MPI_INT, placement.appRank[oldRank], 0,
                 comm_old, MPI_STATUS_IGNORE);
    MPI_Sendrecv(destweights, outdegree, MPI_INT, placement.libRank[oldRank], 0,
                 libdestweights.data(), libdestweights.size(), MPI_INT,
                 placement.appRank[oldRank], 0, comm_old, MPI_STATUS_IGNORE);

    LOG_SPEW("app rank " << placement.appRank[oldRank]);

    int err = libmpi.MPI_Dist_graph_create_adjacent(
        comm_old, libindegree, libsources.data(), libsourceweights.data(),
        liboutdegree, libdestinations.data(), libdestweights.data(), info,
        0 /*we did reordering*/, comm_dist_graph);

    topology::cache_communicator(*comm_dist_graph);
    topology::cache_placement(*comm_dist_graph, placement);

    return err;
  }

  return libmpi.MPI_Dist_graph_create_adjacent(
      ARGS_MPI_Dist_graph_create_adjacent);
}
