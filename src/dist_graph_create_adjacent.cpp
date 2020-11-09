#include "env.hpp"
#include "logging.hpp"
#include "partition.hpp"
#include "symbols.hpp"
#include "topology.hpp"

#include <nvToolsExt.h>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <tuple>

// gather sendbuf from each rank into recvbuf at the root
// at the root recvbuf will be resized appropriately
int TEMPI_Gatherv(const std::vector<int> &sendbuf, std::vector<int> &recvbuf,
                  int root, MPI_Comm comm) {

  int err = MPI_SUCCESS;

  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);

  // get the counts from each rank
  std::vector<int> counts(size, -1);
  {
    int tmp = sendbuf.size();
    err = MPI_Gather(&tmp, 1, MPI_INT, counts.data(), 1, MPI_INT, root, comm);
  }

  if (MPI_SUCCESS != err) {
    return err;
  }

  std::vector<int> displs(size, 0);
  if (root == rank) {
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + counts[i - 1];
    }
    recvbuf.resize(displs[size - 1] + counts[size - 1]);
  }

  err = MPI_Gatherv(sendbuf.data(), sendbuf.size(), MPI_INT, recvbuf.data(),
                    counts.data(), displs.data(), MPI_INT, root, comm);

  return err;
}

extern "C" int
MPI_Dist_graph_create_adjacent(PARAMS_MPI_Dist_graph_create_adjacent) {
  if (environment::noTempi) {
    return libmpi.MPI_Dist_graph_create_adjacent(
        ARGS_MPI_Dist_graph_create_adjacent);
  }

  if (PlacementMethod::NONE == environment::placement) {
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

    } else if (reorder) {

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
      TEMPI_Gatherv(edgeSrc, edgeSrc, 0, comm_old);
      TEMPI_Gatherv(edgeDst, edgeDst, 0, comm_old);
      TEMPI_Gatherv(weight, weight, 0, comm_old);

      // build CSR on root node
      if (0 == oldRank) {
        std::vector<std::tuple<int, int, int>> edges; // src dst weight
        for (size_t i = 0; i < edgeSrc.size(); ++i) {
          edges.push_back(std::make_tuple(edgeSrc[i], edgeDst[i], weight[i]));
        }
        LOG_SPEW("built raw edge list");

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
            // don't want to skip the next element when ++i
            --i;
          }
        }
        LOG_SPEW("deleted self edges");

        std::sort(edges.begin(), edges.end());
        LOG_SPEW("sorted");

        // delete duplicated edges
        for (int64_t i = 0; i < int64_t(edges.size()) - 1; ++i) {

          // find the position of the back edge
          auto lb =
              std::lower_bound(edges.begin() + i + 1, edges.end(), edges[i]);
          auto ub =
              std::upper_bound(edges.begin() + i + 1, edges.end(), edges[i]);
          if (ub != lb) {
            edges.erase(lb, ub);
          }
        }
        LOG_SPEW("delete duplicate edges");

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

        // add missing back-edges.
        // here we halve each of them, since they will be added together in a
        // later step comparator ignoring weight this can introduce edges with 0
        // weight, so round up the half
        auto ignore_weight = [](const std::tuple<int, int, int> &a,
                                const std::tuple<int, int, int> &b) {
          return std::make_pair(std::get<0>(a), std::get<1>(a)) <
                 std::make_pair(std::get<0>(b), std::get<1>(b));
        };
        for (int64_t i = 0; i < int64_t(edges.size()) - 1; ++i) {
          // back edge with the halved weight
          std::tuple<int, int, int> backedge(std::get<1>(edges[i]),
                                             std::get<0>(edges[i]),
                                             std::get<2>(edges[i]) / 2 + 1);

          // find the position of the back edge
          auto lb = std::lower_bound(edges.begin(), edges.end(), backedge,
                                     ignore_weight);

          if (std::get<0>(*lb) == std::get<0>(backedge) &&
              std::get<1>(*lb) == std::get<1>(backedge)) {
            // back edge exists
          } else {
#if 0
            LOG_SPEW("adding back-edge for " << std::get<0>(edges[i]) << " " << std::get<1>(edges[i]));
#endif
            edges.insert(lb, backedge);
            std::get<2>(edges[i]) = std::get<2>(edges[i]) / 2 + 1;
          }
        }
        LOG_SPEW("added missing back edges");

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

        // bidirectional edge weights must be the same for METIS.
        // sum up the two directions
        for (int64_t i = 0; i < int64_t(edges.size()) - 1; ++i) {
          // back edge with matching weight
          std::tuple<int, int, int> backedge(std::get<1>(edges[i]),
                                             std::get<0>(edges[i]),
                                             std::get<2>(edges[i]));

          // find the position of the back edge
          auto lb = std::lower_bound(edges.begin() + i + 1, edges.end(),
                                     backedge, ignore_weight);
          auto ub = std::upper_bound(edges.begin() + i + 1, edges.end(),
                                     backedge, ignore_weight);
          // this indicates there is a back edge.
          // even though all edges have a back-edge, we may be searching only
          // after the last edge, so we already handled it
          if (lb != ub) {
            std::get<2>(edges[i]) += std::get<2>(*lb);
            std::get<2>(*lb) = std::get<2>(edges[i]);
          }
        }
        LOG_SPEW("matched u,v and v,u edge weights");

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
        std::vector<int> xadj;   // adjacency structure (row pointers)
        std::vector<int> adjncy; // adjacency structure (col indices)
        std::vector<int> adjwgt; // weight of edges
        int rp = 0;
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
          std::string s, t, u;
          for (int e : xadj) {
            s += std::to_string(e) + " ";
          }
          for (int e : adjncy) {
            t += std::to_string(e) + " ";
          }
          for (int e : adjwgt) {
            u += std::to_string(e) + " ";
          }
          LOG_DEBUG("adjwgt: (" << adjwgt.size() << ") " << s);
          LOG_DEBUG("adjncy: (" << adjncy.size() << ") " << t);
          LOG_DEBUG("xadj: (" << xadj.size() << ") " << u);
        }

        partition::Result result;

#ifdef TEMPI_ENABLE_METIS
        if (PlacementMethod::METIS == environment::placement) {
          result = partition::partition_metis(numNodes, xadj, adjncy, adjwgt);
        }
#endif
#ifdef TEMPI_ENABLE_KAHIP
        if (PlacementMethod::KAHIP == environment::placement) {
          result = partition::partition_kahip(numNodes, xadj, adjncy, adjwgt);
        }
#endif

        part = result.part;

        {
          std::string s;
          for (int i : part) {
            s += std::to_string(i) + " ";
          }
          LOG_DEBUG("objective= " << result.objective << " part=" << s);
        }

        if (!partition::is_balanced(result)) {
          LOG_ERROR("partition is not balanced.");
          MPI_Finalize();
          exit(1);
        }

      } // 0 == graphRank

      // broadcast the partition assignment to all nodes
      MPI_Bcast(part.data(), part.size(), MPI_INT, 0, comm_old);
    }

#if TEMPI_OUTPUT_LEVEL >= 4
    for (int r = 0; r < oldSize; ++r) {
      if (r == oldRank) {
        MPI_Barrier(comm_old);
        std::string s;
        for (int e : part) {
          s += std::to_string(e) + " ";
        }
        LOG_SPEW("node assignment " << s);
      }
      MPI_Barrier(comm_old);
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

#if TEMPI_OUTPUT_LEVEL >= 4
    for (int r = 0; r < oldSize; ++r) {
      if (r == oldRank) {
        MPI_Barrier(comm_old);
        std::string s, t;
        for (int e : placement.appRank) {
          s += std::to_string(e) + " ";
        }
        for (int e : placement.libRank) {
          t += std::to_string(e) + " ";
        }
        LOG_SPEW("appRank=" << s << " libRank=" << t);
      }
      MPI_Barrier(comm_old);
    }
#endif

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
    LOG_SPEW("send to " << placement.libRank[oldRank] << " recv from "
                        << placement.appRank[oldRank]);
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

#if TEMPI_OUTPUT_LEVEL >= 4 && 1
    {
      std::string s, t;
      for (auto &e : libsources) {
        s += std::to_string(e) + " ";
      }
      for (auto &e : libdestinations) {
        t += std::to_string(e) + " ";
      }
      for (int r = 0; r < oldSize; ++r) {
        if (r == oldRank) {
          MPI_Barrier(comm_old);
          LOG_SPEW("libsources=" << s << " libdestinations=" << t);
          MPI_Barrier(comm_old);
        }
      }
    }
#endif

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
