#include "logging.hpp"

#include <mpi.h>
#include <cuda_runtime.h>

#include <dlfcn.h>

#include <vector>

extern "C" int MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest,
                         int tag, MPI_Comm comm, MPI_Request *request)
{
    LOG_DEBUG("MPI_Isend");

    // find the underlying MPI call
    typedef int (*Func_MPI_Isend)(const void *buf, int count, MPI_Datatype datatype, int dest,
                                  int tag, MPI_Comm comm, MPI_Request *request);
    static Func_MPI_Isend fn = nullptr;
    if (!fn)
    {
        fn = reinterpret_cast<Func_MPI_Isend>(dlsym(RTLD_NEXT, "MPI_Isend"));
    }

    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, buf);
    if (cudaMemoryTypeUnregistered == attr.type)
    {
        LOG_DEBUG("unregistered host");
        return fn(buf, count, datatype, dest, tag, comm, request);
    }
    else if (cudaMemoryTypeHost == attr.type)
    {
        LOG_DEBUG("registered host");
        return fn(buf, count, datatype, dest, tag, comm, request);
    }
    else if (cudaMemoryTypeDevice == attr.type)
    {
        LOG_DEBUG("device");
    }
    else if (cudaMemoryTypeManaged == attr.type)
    {
        LOG_DEBUG("managed");
        return fn(buf, count, datatype, dest, tag, comm, request);
    }

    // decode a possible strided type
    // https://www.mpi-forum.org/docs/mpi-2.0/mpi-20-html/node161.htm
    // MPI_Type_get_envelope()
    // MPI_Type_get_contents()
    // MPI_Type_vector(int count, int blocklength, int stride,MPI_Datatype old_type,MPI_Datatype *newtype_p)
    {
        int numIntegers;
        int numAddresses;
        int numDatatypes;
        int combiner;
        MPI_Type_get_envelope(datatype, &numIntegers, &numAddresses, &numDatatypes, &combiner);
        std::vector<int> integers(numIntegers);
        std::vector<MPI_Aint> addresses(numAddresses);
        std::vector<MPI_Datatype> datatypes(numDatatypes);

        if (MPI_COMBINER_VECTOR == combiner)
        {
            LOG_DEBUG("a vector type!");
            MPI_Type_get_contents(datatype, integers.size(), addresses.size(), datatypes.size(), integers.data(), addresses.data(), datatypes.data());
        }
        else if (MPI_COMBINER_NAMED == combiner)
        {
            LOG_DEBUG("a named type!");
        }
    }

    return MPI_SUCCESS;
}

extern "C" int MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source,
                         int tag, MPI_Comm comm, MPI_Request *request)
{
    LOG_DEBUG("MPI_Irecv");

    // find the underlying MPI call
    typedef int (*Func_MPI_Irecv)(void *buf, int count, MPI_Datatype datatype, int source,
                                  int tag, MPI_Comm comm, MPI_Request *request);
    static Func_MPI_Irecv fn = nullptr;
    if (!fn)
    {
        fn = reinterpret_cast<Func_MPI_Irecv>(dlsym(RTLD_NEXT, "MPI_Irecv"));
    }

    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, buf);
    if (cudaMemoryTypeUnregistered == attr.type)
    {
        LOG_DEBUG("unregistered host");
        return fn(buf, count, datatype, source, tag, comm, request);
    }
    else if (cudaMemoryTypeHost == attr.type)
    {
        LOG_DEBUG("registered host");
        return fn(buf, count, datatype, source, tag, comm, request);
    }
    else if (cudaMemoryTypeDevice == attr.type)
    {
        LOG_DEBUG("device");
    }
    else if (cudaMemoryTypeManaged == attr.type)
    {
        LOG_DEBUG("managed");
        return fn(buf, count, datatype, source, tag, comm, request);
    }

    return MPI_SUCCESS;
}
