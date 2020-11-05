#pragma once

#include <cstdint>
#include <string>
#include <vector>

class SquareMat {
private:
  size_t n_;

public:
  std::vector<int> data_;

public:
  SquareMat(int n, int val) : n_(n), data_(n * n, val) {}
  size_t size() const noexcept { return n_; }

  int *operator[](size_t i) noexcept { return &data_[i * n_]; }
  const int *operator[](size_t i) const noexcept { return &data_[i * n_]; }

  int64_t reduce_sum() const noexcept {
    int64_t ret = 0;
    for (int e : data_) {
      ret += e;
    }
    return ret;
  }

  std::string str() const noexcept;

  /* create a `ranks` x `ranks` matrix
     each value will be in [`lb`, `ub`) * `scale`
   */
  static SquareMat make_random(int ranks, int lb, int ub, int scale);

  /* create a `ranks` x `ranks` matrix with `rowNz` in each row.
     each value will be in [`lb`, `ub`) * `scale`
   */
  static SquareMat make_random_sparse(int ranks, int rowNnz, int lb, int ub,
                                      int scale);

  /* create a `ranks` x `ranks` matrix with blocks on the diagonal of size [bLb,
     bUb) each value will be in [`lb`, `ub`) * `scale`
   */
  static SquareMat make_block_diagonal(int ranks, int bLb, int bUb, int lb,
                                       int ub, int scale);

  // return SquareMat[i][j] = mat[p[i]][p[j]]
  static SquareMat make_permutation(const SquareMat &mat,
                                    const std::vector<int> &p);
};
