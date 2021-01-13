//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "../support/squaremat.hpp"
#include "../include/statistics.hpp"

#include <chrono>
#include <string>

namespace BM {
typedef std::chrono::system_clock Clock;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::time_point<Clock, Duration> Time;

/* The result of a benchmark run.

   Each run will do setup once, iterations some number of times, and then
   teardown once
*/
struct Result {
  std::string name;
  Duration setup;
  Duration teardown;
  Statistics iters;
};

class Method {
public:
  virtual const char *name() = 0;
  virtual ~Method() {}
  virtual Result operator()(const SquareMat &mat, const int nIters) = 0;
};

// MPI_Alltoallv
class Method_alltoallv : public Method {
  const char *name() { return "alltoallv"; }
  Result operator()(const SquareMat &mat, const int nIters);
};

// MPI_Isend / MPI_Irecv from a single buffer
class Method_isend_irecv : public Method {
  const char *name() { return "isend_irecv"; }
  Result operator()(const SquareMat &mat, const int nIters);
};

// isend_irecv, but no call for 0 size
class Method_sparse_isend_irecv : public Method {
  const char *name() { return "sparse_isend_irecv"; }
  Result operator()(const SquareMat &mat, const int nIters);
};

// MPI_Dist_graph_create_adjacent and MPI_Neighbor_alltoallv with reorder
class Method_neighbor_alltoallv : public Method {
  const char *name() { return "reorder_neighbor_alltoallv"; }
  Result operator()(const SquareMat &mat, const int nIters);
};

} // namespace BM
