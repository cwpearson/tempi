/* can't use logging in here since PMPI_Init has not been called
*/
#include <mpi.h>

#include <dlfcn.h>

extern "C" int MPI_Init(int *argc, char ***argv)
{
    typedef int (*Func_MPI_Init)(int *argc, char ***argv);
    static Func_MPI_Init fn = nullptr;

    if (!fn)
    {
        fn = reinterpret_cast<Func_MPI_Init>(dlsym(RTLD_NEXT, "MPI_Init"));
    }
    return fn(argc, argv);
}
