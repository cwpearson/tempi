# TEMPI

Experimental performance enhancmenets for CUDA+MPI codes.
Some improvements require no code modification, only linking the scampi library before your true MPI library.
Other improvements also require `#include tempi/mpi-ext.h` to utilize.

## Quick Start

```
mkdir build
cd build
cmake ..
make
make test
```

Add the library to your link step before the underlying MPI library.

```cmake
add_subdirectory(tempi)
target_link_libraries(my-exe PRIVATE pmpi)
target_link_libraries(my-exe PRIVATE ${MPI_CXX_LIBRARIES})
```

## Features

Performance fixes for CUDA+MPI code that requires no or minimal changes:
- [x] `MPI_Pack` on 3D strided data types

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
As we do not extend the MPI interface, there is no include files to add to your code.

* The API overrides are defined in `src/*.cpp`.
* Most of the internal heavy lifting is done by `include/` and `src/internal`.
  * Reading environment variable configuration is in `include/env.hpp` and `src/internal/env.cpp`.
* Testing code is in `test/`
  * Testing support code is in `test/support`.

## Knobs

The system can be controlled by environment variables.
Setting the corresponding variable to any value (even empty) will change behavior.

|Environment Variable|Effect when Set|
|-|-|
|`TEMPI_NO_PACK`|Defer to library `MPI_Pack`|
|`TEMPI_NO_TYPE_COMMIT`|Don't analyze MPI types for allowable optimizations.|

to unset an environment variable in bash: `unset <VAR>`


## Project Name Ideas

* HEMPI heterogeneous experiments for MPI
* SCAMPI: supercomputing accelerating MPI
* TEMPI: Topology-Enhanced MPI
* TEMPI: Topology experiments for MPI
* MPIsces: MPI supercomputing enhancements
* MPIne: MPI node enhanced
* impish: Improved MPI for system heterogeneous
* MPIsa
* MPIca
* MPIano
* MPIed
* MPIne
* MPIxy

* available
* collisions
  * tempi, (asyc metacomputing in MPI)
  * scampi
    * Scala language MPI
    * Scalable Copy Accelerated by MPI
  * IMPI
    * Gropp, Interoperable MPI