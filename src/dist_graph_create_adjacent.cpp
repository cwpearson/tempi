#include "env.hpp"
#include "logging.hpp"
#include "partition.hpp"
#include "symbols.hpp"
#include "topology.hpp"

#include <metis.h>

#include <algorithm>
#include <cassert>
#include <numeric>

extern "C" int
MPI_Dist_graph_create_adjacent(PARAMS_MPI_Dist_graph_create_adjacent) {
  if (environment::noTempi) {
    return libmpi.MPI_Dist_graph_create_adjacent(
        ARGS_MPI_Dist_graph_create_adjacent);
  }

  /* call the underlying impl to actually create a communicator

     This also places the ranks on the nodes (the "library rank")
     We may be able to determine a better mapping between ranks and nodes.
     If so, we will relabel all the ranks, and present a different rank number
     to the application (the "application rank");

     IF ranks 0 and 1 should be on the same node, we will map ranks 0 and 1 to
     ranks I and J, where I and J are actually on the same node. when rank I
     calls MPI_Comm_rank, they will get 0. When rank I calls MPI_Send to rank 1,
     we will call MPI_Send to rank J if these ranks call collectives, their send
     and recv buffers may be out of order, so we will neeed to permute them
     under the hood which will slow things down
  */
  int err = libmpi.MPI_Dist_graph_create_adjacent(
      ARGS_MPI_Dist_graph_create_adjacent);
  int graphSize, graphRank;
  libmpi.MPI_Comm_rank(*comm_dist_graph, &graphRank);
  libmpi.MPI_Comm_size(*comm_dist_graph, &graphSize);

  // get the topology for the new communicator
  topology::cache_communicator(*comm_dist_graph);

  // assign each rank to a random partition
  const size_t numNodes = topology::num_nodes(*comm_dist_graph);

  if (numNodes > 1) {

    // graph partitioning
    if (reorder && Placement::RANDOM == environment::placement) {

      std::vector<int> partAssignment = partition::random(graphSize, numNodes);
#if TEMPI_OUTPUT_LEVEL >= 4
      {
        if (0 == graphRank) {
          std::string s("node assignment app rank: ");
          for (int r : partAssignment) {
            s += std::to_string(r) + " ";
          }
          LOG_SPEW(s);
        }
      }
#endif

      // all the ranks assign to partition 0 will be backed by ranks on node 0
      topology::cache_node_assignment(*comm_dist_graph, partAssignment);

    } else if (reorder && Placement::METIS == environment::placement) {

      // build edgelist on every node
      std::vector<int> edgeSrc, edgeDst, weight;
      for (int i = 0; i < indegree; ++i) {
        edgeSrc.push_back(sources[i]);
        edgeDst.push_back(graphRank);
        weight.push_back(sourceweights[i]);
      }
      for (int i = 0; i < outdegree; ++i) {
        edgeSrc.push_back(graphRank);
        edgeDst.push_back(destinations[i]);
        weight.push_back(destweights[i]);
      }

      // gather edgelist on root node
      if (0 == graphRank) {
        edgeSrc.resize(graphSize);
        edgeDst.resize(graphSize);
        weight.resize(graphSize);
      }

      LOG_SPEW("GATHER");
      MPI_Gather(edgeSrc.data(), edgeSrc.size(), MPI_INT, edgeSrc.data(),
                 graphSize, MPI_INT, 0, *comm_dist_graph);
      MPI_Gather(edgeDst.data(), edgeDst.size(), MPI_INT, edgeDst.data(),
                 graphSize, MPI_INT, 0, *comm_dist_graph);
      MPI_Gather(weight.data(), weight.size(), MPI_INT, weight.data(),
                 graphSize, MPI_INT, 0, *comm_dist_graph);

      // this is the partition assignment that will be computed on rank 0 and
      // shared
      std::vector<idx_t> part(graphSize);

      // build CSR on root node
      if (0 == graphRank) {
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

        // delete duplicate edges
        bool changed = true;
        while (changed) {
          changed = false;
          for (int64_t i = 0; i < int64_t(edges.size()) - 1; ++i) {
            auto lb =
                std::lower_bound(edges.begin() + i + 1, edges.end(), edges[i]);
            auto ub =
                std::lower_bound(edges.begin() + i + 1, edges.end(), edges[i]);
            if (lb != ub) {
              edges.erase(lb, ub);
              changed = true;
              break;
            }
          }
        }

        // build CSR
        std::vector<idx_t> xadj;   // adjacency structure (row pointers)
        std::vector<idx_t> adjncy; // adjacency structure (col indices)
        std::vector<idx_t> adjwgt; // weight of edges
        idx_t rp = 0;
        for (size_t ei = 0; ei < edges.size(); ++ei) {
          for (; rp <= std::get<0>(edges[ei]); ++rp) {
            xadj.push_back(rp);
          }
          adjncy.push_back(std::get<1>(edges[ei]));
          adjwgt.push_back(std::get<2>(edges[ei]));
        }
        for (; rp <= graphSize; ++rp) {
          xadj.push_back(adjncy.size());
        }
        assert(xadj.size() == graphSize);

        // debug output
        {
          std::string s;
          for (idx_t e : xadj) {
            s += std::to_string(e) + " ";
          }
          LOG_SPEW("xadj: " << s);
        }
        {
          std::string s;
          for (idx_t e : adjncy) {
            s += std::to_string(e) + " ";
          }
          LOG_SPEW("adjncy: " << s);
        }
        {
          std::string s;
          for (idx_t e : adjwgt) {
            s += std::to_string(e) + " ";
          }
          LOG_SPEW("adjwgt: " << s);
        }

        idx_t nvtxs = graphSize; // number of vertices
        idx_t ncon = 1; // number of balancing constraints (at least 1) (?)
        idx_t *vwgt = nullptr; // weights of vertices. null for us to make all
                               // vertices the same
        idx_t *vsize =
            nullptr; // size of vertices. null for us to minimize edge cut
        idx_t nparts = numNodes; // number of partitions for the graph
        real_t *tpwgts = nullptr; // equally divided among partitions
        real_t *ubvec = nullptr; // load imbalance tolerance constraint is 1.001
        idx_t options[METIS_NOPTIONS]{};

        // kway options. comment out means 0 default is okay
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_DBGLVL] = 1;
        idx_t objval;

        int metisErr =
            METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), vwgt,
                                vsize, adjwgt.data(), &nparts, tpwgts, ubvec,
                                options, &objval, part.data());
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

        LOG_SPEW("METIS objval=" << objval);
        {
          std::string s;
          for (idx_t i : part) {
            s += std::to_string(i) + " ";
          }
          LOG_SPEW("part=" << s);
        }

      } // 0 == graphRank

      // broadcast the partition assignment to all nodes
      {
        if (sizeof(idx_t) == sizeof(int)) {
          MPI_Bcast(part.data(), part.size(), MPI_INT, 0, *comm_dist_graph);
        } else if (sizeof(idx_t) == sizeof(int64_t)) {
          MPI_Bcast(part.data(), part.size(), MPI_INT64_T, 0, *comm_dist_graph);
        } else {
          LOG_FATAL("unexpected size of idx_t");
        }
      }

      topology::cache_node_assignment(*comm_dist_graph, part);

      MPI_Barrier(*comm_dist_graph);
    }

  } // numNodes > 1

  return err;
}
