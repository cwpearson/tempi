#include "partition.hpp"

#include "logging.hpp"

#include <metis.h>
#include <nvToolsExt.h>

#include <cassert>
#include <vector>

partition::Result partition::partition_metis(const int nParts,
                                             const std::vector<int> &rowPtrs,
                                             const std::vector<int> &colInd,
                                             const std::vector<int> &colVal) {
  static_assert(sizeof(idx_t) == sizeof(int), "wrong metis idx_t");
  assert(rowPtrs.size() > 0);

  partition::Result result;
  result.part = std::vector<int>(rowPtrs.size()-1, -1);

  idx_t nvtxs = rowPtrs.size() - 1; // number of vertices
  idx_t ncon = 1;         // number of balancing constraints (at least 1)
  idx_t *vwgt = nullptr;  // weights of vertices. null for us to make all
                          // vertices the same
  idx_t *vsize = nullptr; // size of vertices. null for us to minimize edge cut
  idx_t nparts = nParts;  // number of partitions for the graph
  real_t *tpwgts = nullptr; // equally divided among partitions
  real_t *ubvec = nullptr;  // load imbalance tolerance constraint is 1.001
  idx_t options[METIS_NOPTIONS]{};
  METIS_SetDefaultOptions(options);
  // options[METIS_OPTION_DBGLVL] = 1;
  idx_t objval;
  idx_t *part = result.part.data();

  // metis won't modify these
  idx_t *xadj = const_cast<idx_t *>(rowPtrs.data());
  idx_t *adjncy = const_cast<idx_t *>(colInd.data());
  idx_t *adjwgt = const_cast<idx_t *>(colVal.data());

  bool success = false;
  // some seeds produce an unbalanced partition
  for (int seed = 0; seed < 20; ++seed) {
    nvtxRangePush("METIS_PartGraphKway");
    options[METIS_OPTION_SEED] =
        seed / 2; // some seeds may produce an unbalanced partition
    int metisErr;
    if (seed % 2) {
      metisErr = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, vwgt,
                                          vsize, adjwgt, &nparts, tpwgts, ubvec,
                                          options, &objval, part);
    } else {
      metisErr =
          METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, vsize, adjwgt,
                              &nparts, tpwgts, ubvec, options, &objval, part);
    }
    nvtxRangePop();

    if (metisErr == METIS_OK) {
      success = partition::is_balanced(result);
      if (success) {
        break;
      } else {
        continue;
      }
    }

    else if (metisErr == METIS_ERROR_INPUT) {
      LOG_FATAL("metis input error");
      break;
    } else if (metisErr == METIS_ERROR_MEMORY) {
      LOG_FATAL("metis memory error");
      break;
    } else if (metisErr == METIS_ERROR) {
      LOG_FATAL("metis other error");
      break;
    } else {
      LOG_FATAL("unexpected METIS error");
      break;
    }
  }

  result.objective = objval;
  return result;
}
