# pmpi



## Quick Start

```
mkdir build
cd build
cmake ..
make test
```

## Objectives

Accelerate iterative CUDA+MPI code with minimal changes to application code

- [ ] 2D/3D strided data with MPI vector datatypes
- [ ] CUDA IPC to bypass intra-node communication
- [ ] cudaGraph API to accelerate handing of strided data

Hints can be used around any MPI call:

```c++
HINT_REPEATED():
MPI_Isend(...);
HINT_CLEAR();
```



## Design

Instead of using the MPI Profiling Interface (PMPI), our functions defer to the next symbol so this library can be chained with libraries that *do* use PMPI.

For example:

```c++
#include <mpi.h>
#include <dlfcn.h>

typedef int (*Func_MPI_Init)(int *argc, char ***argv);
Func_MPI_Init fn = nullptr;

extern "C" int MPI_Init(int *argc, char ***argv)
{
    if (!fn) {
        fn = reinterpret_cast<Func_MPI_Init>(dlsym(RTLD_NEXT, "MPI_Init"));
    }
    return fn(argc, argv);
}
```

instead of 

```c++
#include <mpi.h>

extern "C" int MPI_Init(int *argc, char ***argv)
{
    return PMPI_Init(argc, argv);
}
```

This library should come before any profiling library that uses PMPI in the linker order, otherwise the application will not call these implementations.



## Project Name

* available
* collisions
  * tempi, (asyc metacomputing in MPI)
  * scampi
    * Scala language MPI
    * Scalable Copy Accelerated by MPI
  * IMPI
    * Gropp, Interoperable MPI