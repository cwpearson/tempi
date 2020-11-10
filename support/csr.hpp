#pragma once

#include "squaremat.hpp"

#include <vector>

struct CSR {
  std::vector<int> rowPtr;
  std::vector<int> colInd;
  std::vector<int> colVal;

  CSR(const SquareMat &mat);
};
