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

  MPI_Datatype cube = factory(packExt, allocExt);
  MPI_Type_commit(&cube);

  // size of the buffer for the data to be packed
  size_t srcSize = allocExt.flatten() * count;

  // size of the data when packed
  int packedSize;
  MPI_Pack_size(count, cube, MPI_COMM_WORLD, &packedSize);
  LOG_DEBUG("packedSize=" << packedSize);

  char *src{}, *packTest{}, *packExp{}, *unpackTest{};

  CUDA_RUNTIME(cudaMallocManaged(&src, srcSize));
  CUDA_RUNTIME(cudaMallocManaged(&unpackTest, srcSize));

  // prefetch to host and zero
  CUDA_RUNTIME(cudaMallocManaged(&packTest, packedSize));
  CUDA_RUNTIME(cudaMemPrefetchAsync(packTest, packedSize, -1, kernStream[0]));
  packExp = new char[packedSize];
  for (size_t i = 0; i < packedSize; ++i) {
    ((char *)packTest)[i] = 0;
    ((char *)packExp)[i] = 0;
  }

  // prefetch to host and initialize with byte offset
  CUDA_RUNTIME(cudaMemPrefetchAsync(src, srcSize, -1, kernStream[0]));
  CUDA_RUNTIME(cudaMemPrefetchAsync(unpackTest, srcSize, -1, kernStream[0]));
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
    MPI_Pack(src, count, cube, packExp, packedSize, &positionExp,
             MPI_COMM_WORLD);
  }

  // test
  {
    LOG_DEBUG("TEMPI MPI_Pack...");
    // prefetch to GPU to accelerate kernel
    CUDA_RUNTIME(cudaMemPrefetchAsync(src, srcSize, 0, kernStream[0]));
    CUDA_RUNTIME(cudaMemPrefetchAsync(packTest, packedSize, 0, kernStream[0]));
    environment::noPack = false;
    positionTest = 0;
    MPI_Pack(src, count, cube, packTest, packedSize, &positionTest,
             MPI_COMM_WORLD);
    CUDA_RUNTIME(cudaDeviceSynchronize());
  }

  // prefetch to host to accelerate comparison
  CUDA_RUNTIME(cudaMemPrefetchAsync(packTest, packedSize, -1, kernStream[0]));

  LOG_DEBUG(positionTest << " " << positionExp);
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
  CUDA_RUNTIME(cudaMemPrefetchAsync(packTest, packedSize, 0, kernStream[0]));
  CUDA_RUNTIME(cudaMemPrefetchAsync(unpackTest, srcSize, 0, kernStream[0]));

  // unpack
  {
    environment::noPack = false;
    positionTest = 0;
    positionExp = 0;
    MPI_Unpack(packTest, packedSize, &positionTest, unpackTest, count, cube,
               MPI_COMM_WORLD);
  }

  // prefetch to host for comparison with original
  CUDA_RUNTIME(cudaMemPrefetchAsync(unpackTest, srcSize, -1, kernStream[0]));

  // compare with original
  REQUIRE(0 == memcmp(unpackTest, src, srcSize));

  CUDA_RUNTIME(cudaFree(src));
  CUDA_RUNTIME(cudaFree(packTest));
  CUDA_RUNTIME(cudaFree(unpackTest));
  delete[](char *) packExp;
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
    Dim3 pe(4, 3, 4), ae(200, 200, 200);
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
    Dim3 pe(100, 100, 100), ae(200, 200, 200);
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
    Dim3 pe(100, 100, 100), ae(200, 200, 200);
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