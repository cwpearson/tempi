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
These are the first thing read during MPI_Init(...).
Setting the corresponding variable to any value (even empty) will change behavior.

|Environment Variable|Effect when Set|
|-|-|
|`TEMPI_DISABLE`|Disable all TEMPI behavior. All calls will use underlying library directly|
|`TEMPI_NO_PACK`|Use library `MPI_Pack`|
|`TEMPI_NO_TYPE_COMMIT`|Use library `MPI_Type_commit`. Don't analyze MPI types for allowable optimizations.|

to unset an environment variable in bash: `unset <VAR>`

## OLCF Summit

nsight-systems 2020.3.1.71 can crash with the osrt or mpi profiler turned on. Disable with nsys profile -t cuda,nvtx.

To control the compute mode, use bsub -alloc_flags gpudefault (see olcf.ornl.gov/for-users/system-user-guides/summitdev-quickstart-guide/#gpu-specific-jobs)

To enable GPUDirect, do jsrun --smpiargs="-gpu" ... (see docs.olcf.ornl.gov/systems/summit_user_guide.html, "CUDA-Aware MPI")

Summit wants to find MPI_Init in darshan (`jsrun -E LD_DEBUG=symbols`).


```
symbol=MPI_Init;  lookup in file=bin/bench-mpi-pack [0]
     68381:     symbol=MPI_Init;  lookup in file=/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/gcc-4.8.5/darshan-runtime-3.1.7-cnvxicgf5j4ap64qi6v5gxp67hmrjz43/lib/libdarshan.so [0]
```

Darshan is not explicitly included in the link step when building, so somehow it is injected at runtime.
In any case, we can fix this by `module unload darshan`, so then our MPI_Init happens right after `libpami_cudahook.so`.
Later, the lazy lookup will cause it to happen in `libmpiprofilesupport.so.3` and then `libmpi_ibm.so.3`.


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
