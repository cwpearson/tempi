#include "../support/type.hpp"
#include "env.hpp"
#include "streams.hpp"
#include "test.hpp"

#include "cuda_runtime.hpp"

#include <mpi.h>
#include <nvToolsExt.h>

#include <cstring> // memcmp
#include <sstream>

template <typename CubeFactoryFn>
void test_pack(CubeFactoryFn factory, const Dim3 &packExt, const Dim3 &allocExt,
               const int count) {

  CUDA_RUNTIME(cudaSetDevice(0));

  size_t srcSize = allocExt.flatten() * count;
  size_t dstDize = packExt.flatten() * count;

  void *src = nullptr, *dstTest = nullptr, *dstExpected = nullptr;
  CUDA_RUNTIME(cudaMallocManaged(&src, srcSize));
  CUDA_RUNTIME(cudaMallocManaged(&dstTest, packExt.flatten() * count));
  dstExpected = new char[packExt.flatten() * count];

  // prefetch to host to accelerate initialization
  CUDA_RUNTIME(cudaMemPrefetchAsync(src, srcSize, -1, kernStream[0]));
  for (size_t i = 0; i < srcSize; ++i) {
    ((char *)src)[i] = rand();
  }

  MPI_Datatype cube = factory(packExt, allocExt);
  MPI_Type_commit(&cube);

  int positionTest = 0, positionExpected = 0;

  // expected
  {
    environment::noPack = true;
    positionExpected = 0;
    MPI_Pack(src, 1, cube, dstExpected, packExt.flatten(), &positionExpected,
             MPI_COMM_WORLD);
  }

  // test
  {
    // prefetch to GPU to accelerate kernel
    CUDA_RUNTIME(cudaMemPrefetchAsync(src, srcSize, 0, kernStream[0]));
    environment::noPack = false;
    positionTest = 0;
    MPI_Pack(src, 1, cube, dstTest, packExt.flatten(), &positionTest,
             MPI_COMM_WORLD);
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  // prefetch to host to accelerate comparison
  CUDA_RUNTIME(cudaMemPrefetchAsync(src, srcSize, -1, kernStream[0]));
  REQUIRE(positionTest == positionExpected); // output position is identical
  REQUIRE(0 == memcmp(dstTest, dstExpected,
                      packExt.flatten())); // pack buffer identical

  CUDA_RUNTIME(cudaFree(src));
  CUDA_RUNTIME(cudaFree(dstTest));
  delete[](char *) dstExpected;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  {
    Dim3 pe(2, 3, 4), ae(10, 10, 10);
    int count = 1;
    std::stringstream ss;
    ss << "TEST: "
       << "make_byte_v_hv" << pe << " " << ae << " " << count;
    LOG_INFO(ss.str());
    nvtxRangePush(ss.str().c_str());
    test_pack(make_byte_v_hv, pe, ae, count);
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
    test_pack(make_byte_v_hv, pe, ae, count);
    nvtxRangePop();
  }

  {
    Dim3 pe(4, 3, 4), ae(200,200,200);
    int count = 1;
    std::stringstream ss;
    ss << "TEST: "
       << "make_byte_v_hv" << pe << " " << ae << " " << count;
    LOG_INFO(ss.str());
    nvtxRangePush(ss.str().c_str());
    test_pack(make_byte_v_hv, pe, ae, count);
    nvtxRangePop();
  }

  {
    Dim3 pe(100, 100, 100), ae(200,200,200);
    int count = 1;
    std::stringstream ss;
    ss << "TEST: "
       << "make_byte_v_hv" << pe << " " << ae << " " << count;
    LOG_INFO(ss.str());
    nvtxRangePush(ss.str().c_str());
    test_pack(make_byte_v_hv, pe, ae, count);
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
    test_pack(make_byte_v_hv, pe, ae, count);
    nvtxRangePop();
  }

  {
    Dim3 pe(100, 100, 100), ae(200,200,200);
    int count = 3;
    std::stringstream ss;
    ss << "TEST: "
       << "make_byte_v_hv" << pe << " " << ae << " " << count;
    LOG_INFO(ss.str());
    nvtxRangePush(ss.str().c_str());
    test_pack(make_byte_v_hv, pe, ae, count);
    nvtxRangePop();
  }

  MPI_Finalize();
  return 0;
}