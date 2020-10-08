#include "type.hpp"


MPI_Datatype make_vn_hv_hv(const Dim3 copyExt, // bytes
                           const Dim3 allocExt // bytes
) {
  MPI_Datatype rowType = {};
  MPI_Datatype planeType = {};
  MPI_Datatype fullType = {};
  {
    {
      {
        // number of blocks
        int count = copyExt.x;
        // number of elements in each block
        int blocklength = 1;
        // number of elements between the start of each block
        const int stride = 1;
        MPI_Type_vector(count, blocklength, stride, MPI_BYTE, &rowType);
      }
      int count = copyExt.y;
      int blocklength = 1;
      // bytes between start of each block
      const int stride = allocExt.x;
      MPI_Type_create_hvector(count, blocklength, stride, rowType, &planeType);
    }
    int count = copyExt.z;
    int blocklength = 1;
    // bytes between start of each block
    const int stride = allocExt.x * allocExt.y;
    MPI_Type_create_hvector(count, blocklength, stride, planeType, &fullType);
  }

  return fullType;
}


MPI_Datatype make_v1_hv_hv(const Dim3 copyExt, // bytes
                           const Dim3 allocExt // bytes
) {
  MPI_Datatype rowType = {};
  MPI_Datatype planeType = {};
  MPI_Datatype fullType = {};
  {
    {
      {
        // number of blocks
        int count = 1;
        // number of elements in each block
        int blocklength = copyExt.x;
        // number of elements between the start of each block
        const int stride = allocExt.x;
        MPI_Type_vector(count, blocklength, stride, MPI_BYTE, &rowType);
      }
      int count = copyExt.y;
      int blocklength = 1;
      // bytes between start of each block
      const int stride = allocExt.x;
      MPI_Type_create_hvector(count, blocklength, stride, rowType, &planeType);
    }
    int count = copyExt.z;
    int blocklength = 1;
    // bytes between start of each block
    const int stride = allocExt.x * allocExt.y;
    MPI_Type_create_hvector(count, blocklength, stride, planeType, &fullType);
  }

  return fullType;
}


MPI_Datatype make_v_hv(const Dim3 copyExt, const Dim3 allocExt) {
  MPI_Datatype planeType = {};
  MPI_Datatype fullType = {};
  {
    {
      // number of blocks
      int count = copyExt.y;
      // number of elements in each block
      int blocklength = copyExt.x;
      // elements between start of each block
      const int stride = allocExt.x;
      MPI_Type_vector(count, blocklength, stride, MPI_BYTE, &planeType);
    }
    int count = copyExt.z;
    int blocklength = 1;
    // bytes between start of each block
    const int stride = allocExt.x * allocExt.y;
    MPI_Type_create_hvector(count, blocklength, stride, planeType, &fullType);
  }

  return fullType;
}


MPI_Datatype make_hi(const Dim3 copyExt, const Dim3 allocExt) {

  MPI_Datatype fullType = {};
  // z*y rows
  const int count = copyExt.z * copyExt.y;

  // byte offset of each row
  MPI_Aint *const displacements = new MPI_Aint[count];
  for (int64_t z = 0; z < copyExt.z; ++z) {
    for (int64_t y = 0; y < copyExt.y; ++y) {
      MPI_Aint bo = z * allocExt.y * allocExt.x + y * allocExt.x;
      // std::cout << bo << "\n";
      displacements[z * copyExt.y + y] = bo;
    }
  }
  // each row is the same length
  int *const blocklengths = new int[count];
  for (int i = 0; i < count; ++i) {
    blocklengths[i] = copyExt.x;
  }

  MPI_Type_create_hindexed(count, blocklengths, displacements, MPI_BYTE,
                           &fullType);
  return fullType;
}


MPI_Datatype make_hib(const Dim3 copyExt, const Dim3 allocExt) {
  MPI_Datatype fullType = {};
  // z*y rows
  const int count = copyExt.z * copyExt.y;
  const int blocklength = copyExt.x;

  // byte offset of each row
  MPI_Aint *const displacements = new MPI_Aint[count];
  for (int64_t z = 0; z < copyExt.z; ++z) {
    for (int64_t y = 0; y < copyExt.y; ++y) {
      MPI_Aint bo = z * allocExt.y * allocExt.x + y * allocExt.x;
      // std::cout << bo << "\n";
      displacements[z * copyExt.y + y] = bo;
    }
  }

  MPI_Type_create_hindexed_block(count, blocklength, displacements, MPI_BYTE,
                                 &fullType);
  return fullType;
}