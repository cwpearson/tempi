#include <cuda_runtime.h>
#include <mpi.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

inline void checkCuda(cudaError_t result, const char *file, const int line) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s:%d: CUDA Runtime Error %d: %s\n", file, line,
            int(result), cudaGetErrorString(result));
    exit(-1);
  }
}

#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);

cudaExtent div_cudaExtent(cudaExtent &a, cudaExtent &b) {
  cudaExtent c;
  c.width = a.width / b.width;
  c.height = a.height / b.height;
  c.depth = a.depth / b.depth;
  return c;
}

void to_3d(int *d3, int d1, const int dim[3]) {
  d3[0] = d1 % dim[0];
  d1 /= dim[0];
  d3[1] = d1 % dim[1];
  d1 /= dim[1];
  d3[2] = d1;
  assert(d3[0] < dim[0]);
  assert(d3[1] < dim[1]);
  assert(d3[2] < dim[2]);
}

void to_1d(int *d1, const int d3[3], const int dim[3]) {
  *d1 = d3[0];
  *d1 += d3[1] * dim[0];
  *d1 += d3[2] * dim[1] * dim[0];
  assert(d1 < dim3[0] * dim3[1] * dim3[2]);
}

/* return the halo size in bytes for a region of size `ext`, with `radius`, for
 * a send in direction `dir`
 */
size_t halo_size(const int radius, cudaExtent ext, const int dir[3]) {
  size_t ret = 0;

  return ret;
}

struct BenchResult {};

BenchResult bench(MPI_Comm comm, int nquants, int radius) {

  BenchResult result;

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // distributed extent
  cudaExtent distExt = make_cudaExtent(512, 512, 512);
  int dims[3];
  if (1 == size) {
    dims[0] = 1;
    dims[1] = 1;
    dims[2] = 1;
  } else if (2 == size) {
    dims[0] = 2;
    dims[1] = 1;
    dims[2] = 1;
  } else if (4 == size) {
    dims[0] = 2;
    dims[1] = 2;
    dims[2] = 1;
  } else {
    std::cerr << "bad dims\n";
    exit(1);
  }

  // local extent
  cudaExtent locExt;
  locExt.width = distExt.width / dims[0];
  locExt.height = distExt.height / dims[1];
  locExt.depth = distExt.depth / dims[2];

  // my coordinates in the distributed space
  int mycoord[3];
  to_3d(mycoord, rank, dims);

  // determine the ranks of my neighbors and how much data I exchange with them
  std::vector<int> nbrs;
  std::vector<size_t> nbrBufSize;
  for (int dz = -1; dz <= 1; ++dz) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (!dx && !dy && !dz) {
          continue;
        }

        int nbrcoord[3];
        nbrcoord[0] = mycoord[0] + dx;
        nbrcoord[1] = mycoord[1] + dy;
        nbrcoord[2] = mycoord[2] + dz;
        while (nbrcoord[0] < 0) {
          nbrcoord[0] += dims[0];
        }
        while (nbrcoord[1] < 0) {
          nbrcoord[1] += dims[1];
        }
        while (nbrcoord[2] < 0) {
          nbrcoord[2] += dims[2];
        }
        nbrcoord[0] %= dims[0];
        nbrcoord[1] %= dims[1];
        nbrcoord[2] %= dims[2];

        int nbrRank;
        to_1d(&nbrRank, nbrcoord, dims);
        nbrs.push_back(nbrRank);
        int dir[3];
        dir[0] = dx;
        dir[1] = dy;
        dir[2] = dz;
        size_t bufSize = halo_size(radius, locExt, dir);
        nbrBufSize.push_back(bufSize);
      }
    }
  }

  // print neighbors
  for (int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == r) {
      std::cout << "rank " << rank << " nbrs= ";
      for (int n : nbrs) {
        std::cout << n << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << std::flush;

  // print volumes
  for (int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == r) {
      std::cout << "rank " << rank << " bufSize= ";
      for (size_t n : nbrBufSize) {
        std::cout << n << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << std::flush;

// create topology
#if 0
{
MPI_Comm comm_old = MPI_COMM_WORLD;
 intindegree, constintsources[], constintsourceweights[],
      intoutdegree, constintdestinations[], constintdestweights[], MPI_Infoinfo,
      intreorder, MPI_Comm * comm_dist_graph

  MPI_Dist_graph_create_adjacent(
      comm_old, indegree, sources, sourceweights,
      outdegree, destinations, destweights, info,
      reorder, comm_dist_graph)
}
#endif

#if 0



  // make my own allocation
  std::vector<cudaPitchedPtr> data(nquants);
  for (auto &ptr : data) {
    CUDA_RUNTIME(cudaMalloc3D(&ptr, rankSize));
  }

  for (int r = 0; r < size; ++r) {
    MPI_Barrier(cart);
    if (rank == r) {
      int neighbors int dst;
      MPI_Cart_shift(cart, 0 /*dir*/, 1 /*displ*/, &src, &dst);
      std::cout << "rank " << rank << " +0: ";
      if (src >= 0)
        std::cout << src;
      else
        std::cout << "x";
      std::cout << " -> " << rank << " -> ";
      if (dst >= 0)
        std::cout << dst;
      else
        std::cout << "x";
      std::cout << "\n";
    }
  }
  std::cout << std::flush;

  /* what is the order of ranks
  source and dest for disp=1 for each dimension?

  we'll use a single type to describe each halo direction, so the send count
  will always be 0 (if no nbr) or 1
  */
  int sendcounts[3 * 2]{0};
  int recvcounts[3 * 2]{0};
  for (int dir = 0; dir < 3; ++dir) {
    int src, dst;
    MPI_Cart_shift(cart, dir, 1 /*displ*/, &src, &dst);
    // set send count to 1 if a neighbor exists
    if (src >= 0) {
      recvcounts[dir * 2] = 1;
      sendcounts[dir * 2] = 1;
    }
    if (dst >= 0) {
      recvcounts[dir * 2 + 1] = 1;
      sendcounts[dir * 2 + 1] = 1;
    }
  }

  for (int r = 0; r < size; ++r) {
    MPI_Barrier(cart);
    if (rank == r) {
      std::cout << "rank " << rank << " sendcounts=";
      for (int i = 0; i < 3 * 2; ++i) {
        std::cout << sendcounts[i] << " ";
      }
      std::cout << std::endl;
    }
  }
  std::cout << std::flush;

#if 0
  MPI_Neighbor_alltoallw(
      const void *sendbuf, const int sendcounts[], const MPI_Aint sdispls[],
      const MPI_Datatype sendtypes[], void *recvbuf, const int recvcounts[],
      const MPI_Aint rdispls[], const MPI_Datatype recvtypes[], MPI_Comm comm)
#endif

  MPI_Comm_free(&cart);

  // free device allocations
  for (auto &ptr : data) {
    CUDA_RUNTIME(cudaFree(ptr.ptr));
  }

#endif
  return result;
}

int main(void) {
  MPI_Init(nullptr, nullptr);
  bench(MPI_COMM_WORLD, 2, 2);
  MPI_Finalize();
}