//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

/* tests to make sure various types are equivalent
 */

#include <mpi.h>

#include "../support/type.hpp"

#include "test.hpp"

#include <vector>

int main(int argc, char **argv) {

  setenv("TEMPI_DISABLE", "", true);

  MPI_Init(&argc, &argv);

  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (1 != size) {
    std::cerr << "ERROR: requires exactly 1 rank\n";
    exit(1);
  }

  {
    std::vector<MPI_Datatype (*)(int)> factories{
        make_contiguous_byte_v1, make_contiguous_byte_vn,
        make_contiguous_subarray, make_contiguous_contiguous};

    std::vector<int> exts{1, 2, 3, 4};

    for (int ext : exts) {
      // create types
      std::vector<MPI_Datatype> types;
      for (auto factory : factories) {
        MPI_Datatype ty = factory(ext);
        MPI_Type_commit(&ty);
        types.push_back(ty);
      }

      // all types should represent the same number of bytes
      {
        int expSize;
        MPI_Aint expLb, expExtent;
        MPI_Type_size(types[0], &expSize);
        MPI_Type_get_extent(types[0], &expLb, &expExtent);
        for (MPI_Datatype ty : types) {
          int actSize;
          MPI_Aint actLb, actExtent;
          MPI_Type_size(ty, &actSize);
          MPI_Type_get_extent(ty, &actLb, &actExtent);
          REQUIRE(expSize == actSize);
          REQUIRE(expLb == actLb);
          REQUIRE(expExtent == actExtent);
        }
      }
    }
  }

  std::vector<Dim3> allocExts = {Dim3(256, 512, 1024)};
  std::vector<Dim3> copyExts = {Dim3(100, 13, 47)};
  std::vector<Dim3> copyOffs = {Dim3(0, 0, 0)};

  for (Dim3 allocExt : allocExts) {
    for (Dim3 copyExt : copyExts) {
      for (Dim3 copyOff : copyOffs) {

        std::cerr << "TEST: " << allocExt << " " << copyExt << " " << copyOff
                  << "\n";

        // build types
        std::vector<MPI_Datatype> types;
        {
          std::cerr << "make_hib\n";
          MPI_Datatype ty = make_hib(copyExt, allocExt);
          MPI_Type_commit(&ty);
          types.push_back(ty);
        }

        {
          MPI_Datatype ty = make_hi(copyExt, allocExt);
          MPI_Type_commit(&ty);
          types.push_back(ty);
        }

        {
          MPI_Datatype ty = make_byte_v_hv(copyExt, allocExt);
          MPI_Type_commit(&ty);
          types.push_back(ty);
        }

        {
          MPI_Datatype ty = make_byte_v1_hv_hv(copyExt, allocExt);
          MPI_Type_commit(&ty);
          types.push_back(ty);
        }

        {
          MPI_Datatype ty = make_byte_vn_hv_hv(copyExt, allocExt);
          MPI_Type_commit(&ty);
          types.push_back(ty);
        }

        // all types should represent the same number of bytes
        {
          int expected;
          MPI_Type_size(types[0], &expected);
          for (MPI_Datatype ty : types) {
            int actual;
            MPI_Type_size(ty, &actual);
            REQUIRE(expected == actual);
          }
        }

        {
          MPI_Aint expLb, expExtent;
          MPI_Type_get_extent(types[0], &expLb, &expExtent);
          std::cerr << expLb << " " << expExtent << "\n";
          for (MPI_Datatype ty : types) {
            MPI_Aint actLb, actExtent;
            MPI_Type_get_extent(types[0], &actLb, &actExtent);
            REQUIRE(expLb == actLb);
            REQUIRE(expExtent == actExtent);
          }
        }

        // all types should pack the same values
        {
          int incount = 1;
          int packSize;
          MPI_Pack_size(incount, types[0], MPI_COMM_WORLD, &packSize);
          std::cerr << "packSize=" << packSize << "\n";
          std::vector<char> expPacked(packSize, -1);

          std::vector<char> toPack(allocExt.flatten());
          int expPosition = 0;
          MPI_Pack(toPack.data(), incount, types[0], expPacked.data(), packSize,
                   &expPosition, MPI_COMM_WORLD);

          for (size_t i = 1; i < types.size(); ++i) {
            MPI_Datatype ty = types[i];
            std::vector<char> actPacked(packSize, -1);
            int testPosition = 0;
            std::cerr << "MPI_Pack(types[" << i << "])\n";
            MPI_Pack(toPack.data(), incount, types[0], actPacked.data(),
                     packSize, &testPosition, MPI_COMM_WORLD);

            for (size_t i = 0; i < packSize; ++i) {
              REQUIRE(actPacked[i] == expPacked[i]);
            }
            REQUIRE(expPosition == testPosition);
          }
        }
      }
    }
  }

  {
    Dim3 copyExt = {.x = 100, .y = 100, .z = 1};
    Dim3 allocExt = {.x = 100, .y = 100, .z = 100};
    std::cerr << "TEST: v1 hv hv\n";
    MPI_Datatype ty = make_byte_v1_hv_hv(copyExt, allocExt);
    MPI_Type_commit(&ty);
    int size;
    MPI_Type_size(ty, &size);
    REQUIRE(size == copyExt.flatten());
  }

  {
    std::cerr << "TEST: by rows / by cols for MPI equivalence\n";
    MPI_Datatype t1 = make_2d_hv_by_rows(13, 3, 16, 5, 53);
    MPI_Datatype t2 = make_2d_hv_by_cols(13, 3, 16, 5, 53);
    MPI_Type_commit(&t1);
    MPI_Type_commit(&t2);

    {
      int size;
      MPI_Type_size(t1, &size);
      REQUIRE(size == 15 * 13);
      MPI_Type_size(t2, &size);
      REQUIRE(size == 15 * 13);
    }

    {
      MPI_Aint lb, ext;
      MPI_Type_get_extent(t1, &lb, &ext);
      REQUIRE(ext == 53 * 4 + 16 * 2 + 13);
      MPI_Type_get_extent(t2, &lb, &ext);
      REQUIRE(ext == 53 * 4 + 16 * 2 + 13);
    }

    MPI_Type_free(&t1);
    MPI_Type_free(&t2);
  }

  MPI_Finalize();

  unsetenv("TEMPI_DISABLE");

  return 0;
}