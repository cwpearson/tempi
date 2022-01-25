#include "tags.hpp"

#include "logging.hpp"

namespace tags {

struct Tags {
  int neighbor_alltoallw = -1;
} info;

int neighbor_alltoallw(MPI_Comm comm) {
  (void)comm; // FIXME: may depend on comm
  return info.neighbor_alltoallw;
}

void init() {

  // FIXME: this should be determined per communicator
  int ub, flag;
  MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &ub, &flag);
  if (!flag) {
    LOG_WARN("couldn't get MPI_TAG_UB. Assuming >= 32767");
    ub = 32767;
  }
  info.neighbor_alltoallw = ub - 1;

  LOG_SPEW("neighbor_alltoallw tag: " << info.neighbor_alltoallw);
}

void finalize();

} // namespace tags