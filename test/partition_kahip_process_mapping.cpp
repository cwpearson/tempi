//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

/* don't build this right now since process_mapping often returns unbalanced
 * partitions
 */

#include "../include/partition.hpp"
#include "../support/csr.hpp"
#include "../support/squaremat.hpp"
#include "test.hpp"

CSR make_csr_random_sparse(int numNodes, int ranksPerNode, int rowNnz,
                           int seed) {

  SquareMat mat = SquareMat::make_random_sparse(numNodes * ranksPerNode, rowNnz,
                                                1, 10, 1, seed);

  std::cerr << mat.str() << "\n";

  // no self edges
  mat.zero_diagonal();
  // forward and back-edges must exist
  // f/b edges must have same weight
  mat.balance_symmetric();

  std::cerr << mat.str() << "\n";

  CSR csr(mat);
  REQUIRE(csr.rowPtr.size() == mat.size() + 1);
  REQUIRE(csr.colInd.size() == mat.popcount());
  REQUIRE(csr.colVal.size() == mat.popcount());
  return csr;
}

int main(void) {

  // for logging
  MPI_Init(nullptr, nullptr);

  int rowNnz = 1;
  int matSeed = 101;

  for (int numNodes : {1, 2, 4, 8, 16, 32}) {
    for (int ranksPerNode : {1, 2, 6}) {
      std::cerr << "TEST partition_kahip " << numNodes << " " << ranksPerNode
                << " " << matSeed << "\n";
      CSR csr = make_csr_random_sparse(numNodes, ranksPerNode, rowNnz, matSeed);

      std::cerr << "rowPtr: ";
      for (int e : csr.rowPtr) {
        std::cerr << e << " ";
      }
      std::cerr << "\n";

      std::cerr << "colInd: ";
      for (int e : csr.colInd) {
        std::cerr << e << " ";
      }
      std::cerr << "\n";
      std::cerr << "colVal: ";
      for (int e : csr.colVal) {
        std::cerr << e << " ";
      }
      std::cerr << "\n";

      partition::Result result = partition::kahip_process_mapping(
          numNodes, csr.rowPtr, csr.colInd, csr.colVal);

      std::cerr << "part: ";
      for (int e : result.part) {
        std::cerr << e << " ";
      }
      std::cerr << "\n";
      std::cerr << result.num_parts() << "\n";
      std::cerr << result.objective << "\n";

      REQUIRE(result.num_parts() == numNodes * ranksPerNode);
      REQUIRE(partition::is_balanced(result));
    }
  }

  MPI_Finalize();
}