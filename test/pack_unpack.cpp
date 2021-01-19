//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "../support/type.hpp"
#include "env.hpp"
#include "streams.hpp"
#include "test.hpp"

#include "cuda_runtime.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <cstring> // memcmp
#include <sstream>

void test_pack(MPI_Datatype ty, const int count) {

  MPI_Type_commit(&ty);
  MPI_Aint extent;
  {
    MPI_Aint _;
    MPI_Type_get_extent(ty, &_, &extent);
  }

  // size of the buffer for the data to be packed
  size_t srcSize = extent * count;
  LOG_DEBUG("srcSize=" << srcSize);

  // size of the data when packed
  int packedSize;
  MPI_Pack_size(count, ty, MPI_COMM_WORLD, &packedSize);
  LOG_DEBUG("packedSize=" << packedSize);

  char *src{}, *packTest{}, *packExp{}, *unpackTest{};

  CUDA_RUNTIME(cudaMallocManaged(&src, srcSize));
  CUDA_RUNTIME(cudaMallocManaged(&unpackTest, srcSize));

  // prefetch to host and zero
  CUDA_RUNTIME(cudaMallocManaged(&packTest, packedSize));
  CUDA_RUNTIME(cudaMemPrefetchAsync(packTest, packedSize, -1, kernStream));
  packExp = new char[packedSize];
  for (size_t i = 0; i < packedSize; ++i) {
    ((char *)packTest)[i] = 0;
    ((char *)packExp)[i] = 0;
  }

  // prefetch to host and initialize with byte offset
  CUDA_RUNTIME(cudaMemPrefetchAsync(src, srcSize, -1, kernStream));
  CUDA_RUNTIME(cudaMemPrefetchAsync(unpackTest, srcSize, -1, kernStream));
  for (size_t i = 0; i < srcSize; ++i) {
    ((char *)src)[i] = i;
    ((char *)unpackTest)[i] = i;
  }

  int positionTest = 0, positionExp = 0;

  // expected
  {
    LOG_DEBUG("system MPI_Pack...");
    environment::noPack = true;
    positionExp = 0;
    MPI_Pack(src, count, ty, packExp, packedSize, &positionExp, MPI_COMM_WORLD);
    LOG_DEBUG("system MPI_Pack done");
  }

  // test
  {
    LOG_DEBUG("TEMPI MPI_Pack...");
    // prefetch to GPU to accelerate kernel
    CUDA_RUNTIME(cudaMemPrefetchAsync(src, srcSize, 0, kernStream));
    CUDA_RUNTIME(cudaMemPrefetchAsync(packTest, packedSize, 0, kernStream));
    environment::noPack = false;
    positionTest = 0;
    MPI_Pack(src, count, ty, packTest, packedSize, &positionTest,
             MPI_COMM_WORLD);
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  // prefetch to host to accelerate comparison
  CUDA_RUNTIME(cudaMemPrefetchAsync(packTest, packedSize, -1, kernStream));

  LOG_DEBUG("positions: " << positionTest << " " << positionExp);
  REQUIRE(positionTest == positionExp); // output position is identical

  // compare with system MPI pack results
  for (size_t i = 0; i < packedSize; ++i) {
    unsigned char test = packTest[i];
    unsigned char exp = packExp[i];
    if (test != exp) {
      LOG_FATAL("mismatch at " << i << " expected " << (unsigned int)exp
                               << " got " << (unsigned int)test);
    }
  }

  // prefetch to GPU to accelerate unpack
  CUDA_RUNTIME(cudaMemPrefetchAsync(packTest, packedSize, 0, kernStream));
  CUDA_RUNTIME(cudaMemPrefetchAsync(unpackTest, srcSize, 0, kernStream));

  // unpack
  {
    LOG_DEBUG("TEMPI MPI_Unpack...");
    environment::noPack = false;
    positionTest = 0;
    positionExp = 0;
    MPI_Unpack(packTest, packedSize, &positionTest, unpackTest, count, ty,
               MPI_COMM_WORLD);
    LOG_DEBUG("TEMPI MPI_Unpack done");
  }

  // prefetch to host for comparison with original
  CUDA_RUNTIME(cudaMemPrefetchAsync(unpackTest, srcSize, -1, kernStream));

  // compare with original
  REQUIRE(0 == memcmp(unpackTest, src, srcSize));

  CUDA_RUNTIME(cudaFree(src));
  CUDA_RUNTIME(cudaFree(packTest));
  CUDA_RUNTIME(cudaFree(unpackTest));
  delete[](char *) packExp;

  MPI_Type_free(&ty);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // good to test input extent > 2^32 occasionally but it's very slow
#if 0
  {
    int blockLength = 1;
    int stride = 512;
    int count = 8;
    int numBlocks = 1024 * 1024;
    std::stringstream ss;
    ss << "TEST: "
       << "make_2d_byte_vector " << numBlocks << " " << blockLength << " "
       << stride << " " << count;
    LOG_INFO(ss.str());
    nvtxRangePush(ss.str().c_str());
    test_pack(make_2d_byte_vector(numBlocks, blockLength, stride), count);
    nvtxRangePop();
  }
#endif

  /*1D test
   */

  {
    int extent = 10;
    int count = 30;
    std::stringstream ss;
    ss << "TEST: "
       << "make_contiguous_contiguous " << extent << " " << count;
    LOG_INFO(ss.str());
    nvtxRangePush(ss.str().c_str());
    test_pack(make_contiguous_contiguous(extent), count);
    nvtxRangePop();
  }

  /* 2D tests */

  {
    int64_t nb = 2, bl = 3, st = 4;
    for (int count : {1, 2}) {
      std::stringstream ss;
      ss << "TEST: "
         << "make_2d_byte_vector " << nb << " " << bl << " " << st << " "
         << count;
      LOG_INFO(ss.str());
      nvtxRangePush(ss.str().c_str());

      test_pack(make_2d_byte_vector(nb, bl, st), count);
      nvtxRangePop();
    }
  }

  {
    int64_t nb = 2, bl = 3, st = 4;
    for (int count : {1, 2}) {
      std::stringstream ss;
      ss << "TEST: "
         << "make_2d_byte_subarray " << nb << " " << bl << " " << st << " "
         << count;
      LOG_INFO(ss.str());
      nvtxRangePush(ss.str().c_str());

      test_pack(make_2d_byte_subarray(nb, bl, st), count);
      nvtxRangePop();
    }
  }

  /* 3D tests */

  {
    Dim3 pe(2, 3, 4), ae(16, 16, 16), off(1, 1, 4);
    int count = 1;
    std::stringstream ss;
    ss << "TEST: "
       << "make_off_subarray " << pe << " " << ae << " " << off << " " << count;
    LOG_INFO(ss.str());
    nvtxRangePush(ss.str().c_str());
    test_pack(make_off_subarray(pe, ae, off), count);
    nvtxRangePop();
  }

  {
    Dim3 pe(2, 3, 4), ae(16, 16, 16), off(1, 1, 4);
    for (int count : {1}) { // FIXME: 3D packer extent wrong
      std::stringstream ss;
      ss << "TEST: "
         << "make_off_subarray " << pe << " " << ae << " " << off << " "
         << count;
      LOG_INFO(ss.str());
      nvtxRangePush(ss.str().c_str());
      test_pack(make_off_subarray(pe, ae, off), count);
      nvtxRangePop();
    }
  }

  {
    Dim3 pe(2, 3, 4), ae(10, 10, 10);
    int count = 1;
    std::stringstream ss;
    ss << "TEST: "
       << "make_byte_v_hv" << pe << " " << ae << " " << count;
    LOG_INFO(ss.str());
    nvtxRangePush(ss.str().c_str());
    test_pack(make_byte_v_hv(pe, ae), count);
    nvtxRangePop();
  }

  {
    Dim3 pe(2, 3, 4), ae(10, 10, 10);
    int count = 2;
    std::stringstream ss;
    ss << "TEST: "
       << "make_byte_v_hv" << pe << " " << ae << " " << count;
    LOG_INFO(ss.str());
    nvtxRangePush(ss.str().c_str());
    test_pack(make_byte_v_hv(pe, ae), count);
    nvtxRangePop();
  }

  {
    Dim3 pe(10, 10, 1), ae(10, 10, 10);
    int count = 1;
    std::stringstream ss;
    ss << "TEST: "
       << "make_byte_v_hv" << pe << " " << ae << " " << count;
    LOG_INFO(ss.str());
    nvtxRangePush(ss.str().c_str());
    test_pack(make_byte_v_hv(pe, ae), count);
    nvtxRangePop();
  }

  {
    Dim3 pe(4, 3, 4), ae(200, 200, 200);
    int count = 1;
    std::stringstream ss;
    ss << "TEST: "
       << "make_byte_v_hv" << pe << " " << ae << " " << count;
    LOG_INFO(ss.str());
    nvtxRangePush(ss.str().c_str());
    test_pack(make_byte_v_hv(pe, ae), count);
    nvtxRangePop();
  }

  {
    Dim3 pe(100, 100, 100), ae(200, 200, 200);
    int count = 1;
    std::stringstream ss;
    ss << "TEST: "
       << "make_byte_v_hv" << pe << " " << ae << " " << count;
    LOG_INFO(ss.str());
    nvtxRangePush(ss.str().c_str());
    test_pack(make_byte_v_hv(pe, ae), count);
    nvtxRangePop();
  }

  {
    Dim3 pe(100, 100, 100), ae(200, 200, 200);
    int count = 3;
    std::stringstream ss;
    ss << "TEST: "
       << "make_byte_v_hv" << pe << " " << ae << " " << count;
    LOG_INFO(ss.str());
    nvtxRangePush(ss.str().c_str());
    test_pack(make_byte_v_hv(pe, ae), count);
    nvtxRangePop();
  }

  MPI_Finalize();
  return 0;
}