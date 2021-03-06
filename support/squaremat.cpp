#include "squaremat.hpp"

#include <algorithm>
#include <iostream>
#include <random>

void SquareMat::zero_diagonal() {
  for (size_t i = 0; i < n_; ++i) {
    data_[i * n_ + i] = 0;
  }
}

void SquareMat::balance_symmetric() {
  for (size_t i = 0; i < n_; ++i) {
    for (size_t j = i; j < n_; ++j) {
      int total = data_[i * n_ + j] + data_[j * n_ + i];
      data_[i * n_ + j] = total / 2;
      data_[j * n_ + i] = total / 2;
    }
  }
}

std::string SquareMat::str() const noexcept {
  std::string s;

  for (size_t i = 0; i < n_; ++i) {

    for (size_t j = 0; j < n_; ++j) {
      s += std::to_string(data_[i * n_ + j]) + " ";
    }
    // no final newline
    if (i + 1 < n_) {
      s += "\n";
    }
  }
  return s;
}

static SquareMat make_random(int ranks, int lb, int ub, int scale, int seed) {
  srand(seed);
  std::default_random_engine dre(seed);
  SquareMat mat(ranks, 0);
  for (int r = 0; r < ranks; ++r) {
    for (int c = 0; c < ranks; ++c) {
      int val = (lb + rand() % (ub - lb)) * scale;
      mat[r][c] = val;
    }
  }
  return mat;
}

SquareMat SquareMat::make_random_sparse(int ranks, int rowNnz, int lb, int ub,
                                        int scale, int seed) {
  srand(seed);
  std::default_random_engine dre(seed);

  rowNnz = std::min(rowNnz, ranks);

  SquareMat mat(ranks, 0);

  std::vector<size_t> rowInd(ranks);
  std::iota(rowInd.begin(), rowInd.end(), 0);

  for (int r = 0; r < ranks; ++r) {
    // selet this row's nonzeros
    std::vector<size_t> nzs;
    std::shuffle(rowInd.begin(), rowInd.end(), dre);

    for (size_t i = 0; i < rowNnz; ++i) {
      int val = (lb + rand() % (ub - lb)) * scale;
      mat[r][rowInd[i]] = val;
    }
  }
  return mat;
}

SquareMat SquareMat::make_block_diagonal(int ranks, int bLb, int bUb, int lb,
                                         int ub, int scale) {

  // int rank;
  // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int SEED = 101;
  srand(SEED);

  SquareMat mat(ranks, 0);

  // diagonal position
  for (int d = 0; d < ranks;) {

    // decide block size
    int bSz = (bLb + rand() % (bUb - bLb));
    if (d + bSz >= ranks) {
      bSz = ranks - d;
    }

    // fill block
    for (int r = d; r < d + bSz; ++r) {
      for (int c = d; c < d + bSz; ++c) {
        int val = (lb + rand() % (ub - lb)) * scale;
        mat[r][c] = val;
      }
    }

    d += bSz;
  }
  return mat;
}

/* permute the rows and columns of mat according to `p`
 */
SquareMat SquareMat::make_permutation(const SquareMat &mat,
                                      const std::vector<int> &p) {
  SquareMat next = mat;
  for (int i = 0; i < mat.size(); ++i) {
    for (int j = 0; j < mat.size(); ++j) {
      next[i][j] = mat[p[i]][p[j]];
    }
  }
  return next;
}
