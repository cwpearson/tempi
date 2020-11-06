#include "csr.hpp"

#include <cassert>

CSR::CSR(const SquareMat &mat) {
  for (int r = 0; r < mat.size(); ++r) {
    rowPtr.push_back(colInd.size());
    for (int c = 0; c < mat.size(); ++c) {
      if (0 != mat[r][c]) {
        colInd.push_back(c);
        colVal.push_back(mat[r][c]);
      }
    }
  }
  rowPtr.push_back(colInd.size());

#ifndef NDEBUG

  assert(rowPtr.size() == mat.size() + 1 &&
         "CSR rowPtr array should be rows+1");

  assert(colInd.size() == colVal.size() &&
         "CSR should have one value per nonzero");

  for (int e : colVal) {
    assert(e != 0 && "CSR should have no zero values");
  }

#endif
};