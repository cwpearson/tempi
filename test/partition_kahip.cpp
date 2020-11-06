#include "../include/partition.hpp"
#include "../support/csr.hpp"
#include "../support/squaremat.hpp"
#include "test.hpp"

CSR make_csr_random_sparse(int numNodes, int ranksPerNode, int rowNnz,
                           int seed) {

  SquareMat mat = SquareMat::make_random_sparse(numNodes * ranksPerNode, rowNnz,
                                                1, 10, 1, seed);

  // no self edges
  mat.zero_diagonal();
  // forward and back-edges must exist
  // f/b edges must have same weight
  mat.balance_symmetric();

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
    for (int ranksPerNode : {1, 6}) {
      std::cerr << "TEST partition_kahip " << numNodes << " " << ranksPerNode
                << " " << matSeed << "\n";
      CSR csr = make_csr_random_sparse(numNodes, ranksPerNode, rowNnz, matSeed);
      partition::Result result = partition::partition_kahip(
          numNodes, csr.rowPtr, csr.colInd, csr.colVal);
      REQUIRE(result.num_parts() == numNodes);
      REQUIRE(partition::is_balanced(result));

      //   for (int e : result.part) {
      //     std::cerr << e << " ";
      //   }
      //   std::cerr << "\n";
      //   std::cerr << result.num_parts() << "\n";
      //   std::cerr << result.objective << "\n";
    }
  }

  MPI_Finalize();
}