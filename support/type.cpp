#include "type.hpp"

MPI_Datatype make_byte_vn_hv_hv(const Dim3 copyExt, // bytes
                                const Dim3 allocExt // bytes
) {
  MPI_Datatype row{}, plane{}, cube{};
  {
    {
      {
        // number of blocks
        int count = copyExt.x;
        // number of elements in each block
        int blocklength = 1;
        // number of elements between the start of each block
        const int stride = 1;
        MPI_Type_vector(count, blocklength, stride, MPI_BYTE, &row);
      }
      int count = copyExt.y;
      int blocklength = 1;
      // bytes between start of each block
      const int stride = allocExt.x;
      MPI_Type_create_hvector(count, blocklength, stride, row, &plane);
    }
    int count = copyExt.z;
    int blocklength = 1;
    // bytes between start of each block
    const int stride = allocExt.x * allocExt.y;
    MPI_Type_create_hvector(count, blocklength, stride, plane, &cube);
  }

  return cube;
}

MPI_Datatype make_byte_v1_hv_hv(const Dim3 copyExt, // bytes
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

MPI_Datatype make_byte_v_hv(const Dim3 copyExt, const Dim3 allocExt) {
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

MPI_Datatype make_float_v_hv(const Dim3 copyExt, const Dim3 allocExt) {
  MPI_Datatype planeType = {};
  MPI_Datatype fullType = {};
  {
    {
      // number of blocks
      int count = copyExt.y;
      // number of elements in each block
      int blocklength = copyExt.x / 4;
      // elements between start of each block
      const int stride = allocExt.x / 4;
      MPI_Type_vector(count, blocklength, stride, MPI_FLOAT, &planeType);
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
      displacements[z * copyExt.y + y] = bo;
    }
  }

  MPI_Type_create_hindexed_block(count, blocklength, displacements, MPI_BYTE,
                                 &fullType);
  return fullType;
}

MPI_Datatype make_subarray(const Dim3 copyExt, const Dim3 allocExt) {

  int ndims = 3;
  int array_of_sizes[3]{int(allocExt[2]), int(allocExt[1]), int(allocExt[0])};
  int array_of_subsizes[3]{int(copyExt[2]), int(copyExt[1]), int(copyExt[0])};
  int array_of_starts[3]{0, 0, 0};
  int order = MPI_ORDER_C;

  MPI_Datatype cube{};
  MPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes,
                           array_of_starts, order, MPI_BYTE, &cube);
  return cube;
}

MPI_Datatype make_subarray_v(const Dim3 copyExt, const Dim3 allocExt) {

  MPI_Datatype plane{};
  {
    int ndims = 2;
    int array_of_sizes[2]{int(allocExt[1]), int(allocExt[0])};
    int array_of_subsizes[2]{int(copyExt[1]), int(copyExt[0])};
    int array_of_starts[2]{0, 0};
    int order = MPI_ORDER_C;

    MPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes,
                             array_of_starts, order, MPI_BYTE, &plane);
  }
  MPI_Datatype cube{};
  {
    // number of blocks
    int count = int(copyExt[2]);
    // number of elements in each block
    int blocklength = 1;
    // elements between start of each block
    const int stride = 1;
    MPI_Type_vector(count, blocklength, stride, plane, &cube);
  }

  return cube;
}

MPI_Datatype make_off_subarray(const Dim3 copyExt, const Dim3 allocExt,
                               const Dim3 &off) {

  int ndims = 3;
  int array_of_sizes[3]{int(allocExt[0]), int(allocExt[1]), int(allocExt[2])};
  int array_of_subsizes[3]{int(copyExt[0]), int(copyExt[1]), int(copyExt[2])};
  int array_of_starts[3]{int(off[0]), int(off[1]), int(off[2])};

  MPI_Datatype cube{};
  MPI_Type_create_subarray(ndims, array_of_sizes, array_of_subsizes,
                           array_of_starts, MPI_ORDER_C, MPI_BYTE, &cube);
  return cube;
}

/* 2D layouts
 */

MPI_Datatype make_2d_byte_vector(const int64_t numBlocks,
                                 const int64_t blockLength,
                                 const int64_t stride) {
  MPI_Datatype ty{};
  MPI_Type_vector(numBlocks, blockLength, stride, MPI_BYTE, &ty);
  return ty;
}

MPI_Datatype make_2d_byte_hvector(const int64_t numBlocks,
                                  const int64_t blockLength,
                                  const int64_t stride) {
  MPI_Datatype ty{};
  MPI_Type_create_hvector(numBlocks, blockLength, stride, MPI_BYTE, &ty);
  return ty;
}

MPI_Datatype make_2d_byte_subarray(const int64_t numBlocks,
                                   const int64_t blockLength,
                                   const int64_t stride) {
  MPI_Datatype ty{};
  int array_of_sizes[2]{int(numBlocks), int(stride)};
  int array_of_subsizes[2]{int(numBlocks), int(blockLength)};
  int array_of_starts[2]{0, 0};
  MPI_Type_create_subarray(2, array_of_sizes, array_of_subsizes,
                           array_of_starts, MPI_ORDER_C, MPI_BYTE, &ty);
  return ty;
}

// make a 2d type by rows of blocks, then a stack of rows
MPI_Datatype make_2d_hv_by_rows(int blockSize, int c1,
                                int s1, // stride between blocks
                                int c2,
                                int s2 // stride between rows
) {

  MPI_Datatype bytes, row, ret;

  MPI_Type_contiguous(blockSize, MPI_BYTE, &bytes);
  MPI_Type_create_hvector(c1, 1, s1, bytes, &row);
  MPI_Type_create_hvector(c2, 1, s2, row, &ret);

  return ret;
}

// make a 2d type by a column of blocks, and then a stack of columns.
MPI_Datatype make_2d_hv_by_cols(int blockSize, int c1,
                                int s1, // stride between blocks
                                int c2,
                                int s2 // stride between rows
) {

  MPI_Datatype bytes, col, ret;

  MPI_Type_contiguous(blockSize, MPI_BYTE, &bytes);
  MPI_Type_create_hvector(c2, 1, s2, bytes, &col);
  MPI_Type_create_hvector(c1, 1, s1, col, &ret);

  return ret;
}

/*1D layouts
 */

MPI_Datatype make_contiguous_byte_v1(int n) {
  MPI_Datatype ty{};
  int count = n;
  int blocklength = 1;
  const int stride = 1;
  MPI_Type_vector(count, blocklength, stride, MPI_BYTE, &ty);
  return ty;
}
MPI_Datatype make_contiguous_byte_vn(int n) {
  MPI_Datatype ty{};
  int count = 1;
  int blocklength = n;
  const int stride = n;
  MPI_Type_vector(count, blocklength, stride, MPI_BYTE, &ty);
  return ty;
}
MPI_Datatype make_contiguous_subarray(int n) {
  MPI_Datatype ty{};
  int ndims = 1;
  int size_array[]{n};
  int subsize_array[]{n};
  int start_array[]{0};
  MPI_Type_create_subarray(1, size_array, subsize_array, start_array,
                           MPI_ORDER_C, MPI_BYTE, &ty);
  return ty;
}
MPI_Datatype make_contiguous_contiguous(int n) {
  MPI_Datatype ty{};
  int count = n;
  MPI_Type_contiguous(count, MPI_BYTE, &ty);
  return ty;
}