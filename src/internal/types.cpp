#include "types.hpp"

#include "logging.hpp"
#include "packer_stride_1.hpp"
#include "packer_stride_2.hpp"

#include <cassert>

std::map<MPI_Datatype, Type> traverseCache;
/*extern*/ std::map<MPI_Datatype, std::shared_ptr<Packer>> packerCache;

// decode an MPI datatyp
// https://www.mpi-forum.org/docs/mpi-2.0/mpi-20-html/node161.htm
// MPI_Type_get_envelope()
// MPI_Type_get_contents()
Type Type::from_mpi_datatype(MPI_Datatype datatype) {

  LOG_DEBUG("from_mpi_datatype ");

  int numIntegers;
  int numAddresses;
  int numDatatypes;
  int combiner;
  MPI_Type_get_envelope(datatype, &numIntegers, &numAddresses, &numDatatypes,
                        &combiner);
  std::vector<int> integers(numIntegers);
  std::vector<MPI_Aint> addresses(numAddresses);
  std::vector<MPI_Datatype> datatypes(numDatatypes);

  if (MPI_COMBINER_VECTOR == combiner) {

    Type ret;

    MPI_Type_get_contents(datatype, integers.size(), addresses.size(),
                          datatypes.size(), integers.data(), addresses.data(),
                          datatypes.data());

    /*
    MPI_Type_vector(int count, int blocklength, int stride,MPI_Datatype
    old_type,MPI_Datatype *newtype_p)
    */
    assert(integers.size() == 3);
    assert(datatypes.size() == 1);

    VectorData data;
    data.count = integers[0];
    data.elemLength = integers[1];
    data.elemStride = integers[2];
    ret.data = data;

    LOG_DEBUG("vector count=" << data.count
                              << " blockLength=" << data.byteLength
                              << " blockStride=" << data.byteStride
                              << " elemLength=" << data.elemLength
                              << " elemStride=" << data.elemStride);

    Type child = Type::from_mpi_datatype(datatypes[0]);
    ret.children_.push_back(child);
    return ret;

  } else if (MPI_COMBINER_HVECTOR == combiner) {
    Type ret;
    MPI_Type_get_contents(datatype, integers.size(), addresses.size(),
                          datatypes.size(), integers.data(), addresses.data(),
                          datatypes.data());

    /*
    MPI_Type_create_hvector(count, blocklength, stride, planeType, &fullType);
    */
    assert(integers.size() == 2);
    assert(addresses.size() == 1);
    assert(datatypes.size() == 1);

    VectorData data;
    data.count = integers[0];
    data.elemLength = integers[1];
    data.byteStride = addresses[0];
    LOG_SPEW("hvector -> " << data.str());
    ret.data = data;
    Type child = Type::from_mpi_datatype(datatypes[0]);
    ret.children_.push_back(child);
    return ret;
  } else if (MPI_COMBINER_NAMED == combiner) {
    // names types are vector of length 1 with known sizes
    LOG_DEBUG("named type");
    Type ret;
    VectorData data;
    data.count = 1;
    data.elemStride = 1;
    data.elemLength = 1;
    if (MPI_BYTE == datatype) {
      data.byteLength = 1;
      data.byteStride = 1;
    } else if (MPI_FLOAT == datatype) {
      data.byteLength = 4;
      data.byteStride = 4;
    } else {
      LOG_FATAL("unhandled named type");
    }
    LOG_SPEW("named -> " << data.str());
    ret.data = data;
    return ret;
  } else if (MPI_COMBINER_INDEXED_BLOCK == combiner) {
    LOG_DEBUG("indexed_block");
    LOG_WARN("couldn't convert indexed block to structured type");
    return Type();
  } else if (MPI_COMBINER_HINDEXED_BLOCK == combiner) {
    LOG_DEBUG("hindexed_block");
    LOG_WARN("couldn't convert hindexed block to structured type");
    return Type();
  } else if (MPI_COMBINER_HINDEXED == combiner) {
    // http://www.cse.chalmers.se/~mckee/papers/sc03.pdf
    LOG_DEBUG("hindexed");
    LOG_WARN("couldn't convert hindexed to structured type");
    return Type();
  } else if (MPI_COMBINER_CONTIGUOUS == combiner) {
    LOG_DEBUG("contiguous");
    LOG_WARN("couldn't convert contiguous to structured type");
    return Type();
  } else if (MPI_COMBINER_STRUCT == combiner) {
    LOG_DEBUG("struct");
    LOG_WARN("couldn't convert struct to structured type");
    return Type();
  } else if (MPI_COMBINER_SUBARRAY == combiner) {
    LOG_DEBUG("subarray");

    MPI_Type_get_contents(datatype, integers.size(), addresses.size(),
                          datatypes.size(), integers.data(), addresses.data(),
                          datatypes.data());

    Type ret;

    /*
    MPI_Type_create_subarray(int ndims, const int array_of_sizes[], const
int array_of_subsizes[], const int array_of_starts[], int order, MPI_Datatype
oldtype, MPI_Datatype *newtype)
    */

    // num integers will be 1 (ndims) + 3 * ndims (sizes, subsizes, starts) + 1
    // (order)
    assert(numIntegers >= 2);
    const int ndims = integers[0];
    assert(numIntegers == 2 + 3 * ndims);
    const int order = integers[numIntegers - 1];
    if (order != MPI_ORDER_C) {
      LOG_ERROR("unhandled order in subarray type");
      return ret;
    }

    SubarrayData data;
    for (int i = 0; i < ndims; ++i) {
      data.byteLength = -1;
      data.elemSizes.push_back(integers[1 + ndims * 0 + i]);
      data.elemSubsizes.push_back(integers[1 + ndims * 1 + i]);
      data.elemStarts.push_back(integers[1 + ndims * 2 + i]);
      data.byteStrides.push_back(-1);
    }

    LOG_SPEW("subarray -> " << data.str());

    ret.data = data;

    Type child = Type::from_mpi_datatype(datatypes[0]);
    ret.children_.push_back(child);

    return ret;
  }

  // http://mpi.deino.net/mpi_functions/MPI_Type_get_envelope.html
  /*
  MPI_COMBINER_NAMEDa named predefined datatype
  MPI_COMBINER_DUP MPI_TYPE_DUP
  MPI_COMBINER_CONTIGUOUS MPI_TYPE_CONTIGUOUS
  MPI_COMBINER_VECTOR MPI_TYPE_VECTOR
  MPI_COMBINER_HVECTOR_INTEGER MPI_TYPE_HVECTOR from Fortran
  MPI_COMBINER_HVECTOR MPI_TYPE_HVECTOR from C or C++
  and in some case Fortran
  or MPI_TYPE_CREATE_HVECTOR
  MPI_COMBINER_INDEXED MPI_TYPE_INDEXED
  MPI_COMBINER_HINDEXED_INTEGER MPI_TYPE_HINDEXED from Fortran
  MPI_COMBINER_HINDEXED MPI_TYPE_HINDEXED from C or C++
  and in some case Fortran
  or MPI_TYPE_CREATE_HINDEXED
  MPI_COMBINER_INDEXED_BLOCK MPI_TYPE_CREATE_INDEXED_BLOCK
  MPI_COMBINER_STRUCT_INTEGER MPI_TYPE_STRUCT from Fortran
  MPI_COMBINER_STRUCT MPI_TYPE_STRUCT from C or C++
  and in some case Fortran
  or MPI_TYPE_CREATE_STRUCT
  MPI_COMBINER_DARRAY MPI_TYPE_CREATE_DARRAY
  MPI_COMBINER_F90_REAL MPI_TYPE_CREATE_F90_REAL
  MPI_COMBINER_F90_COMPLEX MPI_TYPE_CREATE_F90_COMPLEX
  MPI_COMBINER_F90_INTEGER MPI_TYPE_CREATE_F90_INTEGER
  MPI_COMBINER_RESIZED MPI_TYPE_CREATE_RESIZED


  If combiner is MPI_COMBINER_NAMED then datatype is a named predefined
  datatype.
  */

  // unknown type
  return Type();
}

Type traverse(MPI_Datatype datatype) {
  if (0 != traverseCache.count(datatype)) {
    return traverseCache[datatype];
  } else {
    Type result = Type::from_mpi_datatype(datatype);
    if (Type() != result) {
      traverseCache[datatype] = result;
    }
    return result;
  }
};

/* try to convert all nodes to subarray type
 */
void convert_nodes_to_subarray(Type &type) {

  // convert all child nodes into subarrays
  for (Type &child : type.children()) {
    convert_nodes_to_subarray(child);
  }

  /* generally a vector is the same as a 2D subarray starting at offset 0,
     where the first dimension is the stride and the second dimension is the
     count

     if the count is 1, it's just a 1D subarray of size elemStride and subsize
     elemLength
     if the stride is the same as the blocklength, it's contiguous, so it's also
     a 1D subarray with size of elemStride * count and subsize of
     elemLength*count
     so, both of those cases are equivalent.

     the type may be a basic type with a known byte size (or equivalent), so
     handle that too.

  */
  if (std::holds_alternative<VectorData>(type.data)) {
    const VectorData &vecData = std::get<VectorData>(type.data);

    assert(vecData.count >= 0);

    SubarrayData data;
    data.byteLength = vecData.byteLength;
    // 1D subarray
    if (1 == vecData.count || vecData.elemStride == vecData.elemLength) {
      data.elemStarts = {0};
      data.elemSizes = {vecData.count * vecData.elemStride};
      data.elemSubsizes = {vecData.count * vecData.elemLength};
      data.byteStrides = {vecData.byteStride};

    } else {                    // 2D subarray
      data.elemStarts = {0, 0}; // no offset

      data.elemSubsizes = {vecData.elemLength, vecData.count};
      if (vecData.elemStride >= 0) {
        data.elemSizes = {vecData.elemStride, vecData.count};
        data.byteStrides = {-1, -1};
      } else if (vecData.byteStride >= 0) {
        data.elemSizes = {-1, vecData.count};
        data.byteStrides = {vecData.byteStride, -1};
      } else {
        LOG_FATAL("should know either elemStride or byteStride");
      }

      data.elemSubsizes = {vecData.elemLength, vecData.count};
    }

    LOG_SPEW(vecData.str() << " -> " << data.str());
    type.data = data;
  }
}

/* try to fuse parent/child subarrays into a single higher-dimension subarray

only works if a parent has a single child and both are subarrays
 */
void fuse_subarrays(Type &type) {

  // try to fuse all children (start at the bottom of the tree)
  for (Type &child : type.children()) {
    fuse_subarrays(child);
  }

  // I'm not a subarray so I can't fuse with my children
  if (!std::holds_alternative<SubarrayData>(type.data)) {
    LOG_SPEW("no fuse: not subarray");
    return;
  }

  // if more than one child, can't fuse
  if (type.children().size() != 1) {
    LOG_SPEW("no fuse: need 1 child (have " << type.children().size() << ")");
    return;
  }

  Type &child = type.children()[0];

  // my child is not a subarray so I can't fuse with it
  if (!std::holds_alternative<SubarrayData>(child.data)) {
    LOG_SPEW("no fuse: child not subarray");
    return;
  }

  LOG_SPEW("fuse opportunity found!");

  // at this point, both me and my single child are subarrays
  // nest the child subarray inside the parent subarray
  SubarrayData fused;
  // first dimensions are child data, then parent data
  const SubarrayData &pData = std::get<SubarrayData>(type.data);
  const SubarrayData &cData = std::get<SubarrayData>(child.data);

  LOG_SPEW("fuse: " << cData.str() << " + " << pData.str());

  fused.byteLength = cData.byteLength;
  for (size_t dim = 0; dim < cData.ndims(); ++dim) {
    fused.elemSizes.push_back(cData.elemSizes[dim]);
    fused.elemStarts.push_back(cData.elemStarts[dim]);
    fused.elemSubsizes.push_back(cData.elemSubsizes[dim]);
    fused.byteStrides.push_back(cData.byteStrides[dim]);
  }
  for (size_t dim = 0; dim < pData.ndims(); ++dim) {
    fused.elemSizes.push_back(pData.elemSizes[dim]);
    fused.elemStarts.push_back(pData.elemStarts[dim]);
    fused.elemSubsizes.push_back(pData.elemSubsizes[dim]);
    fused.byteStrides.push_back(pData.byteStrides[dim]);
  }

  // we're going to delete the child, so we'll need to make it's children our
  // own
  std::vector<Type> grandchildren = child.children();

  // delete the child (only 1 child) and replace with granchildren
  type.children() = grandchildren;
  type.data = fused;

  LOG_SPEW("fused into Subarray ndims=" << fused.ndims());

  LOG_SPEW(fused.str());
}

void subarrays_merge_subsize_one_dims(Type &type) {
  // try to merge all children (start at the bottom of the tree)
  for (Type &child : type.children()) {
    subarrays_merge_subsize_one_dims(child);
  }

  // I'm not a subarray so I can't merge
  if (!std::holds_alternative<SubarrayData>(type.data)) {
    LOG_SPEW("merge_subsize_one: not subarray");
    return;
  }

  SubarrayData &data = std::get<SubarrayData>(type.data);

  /* if a dimension has subSize = 1, that means that it represents only a
     single instance of the lower dimension.
     However, it still spaces out instances of the lower dimension

     if we know the stride of the degenerate dimension, we say the upper
     dimension's stride is now that instead.
     if we know the elemSize of the degenerate dimension, we say the lower
     dimension's elemSize is just that much larger


  */

  bool changed = true;
  while (changed) {
    changed = false;
    for (int i = 0; i < data.ndims(); ++i) {
      if (1 == data.elemSubsizes[i]) {

        LOG_SPEW("remove degenerate dim " << i);

        if (data.byteStrides[i] >= 0) {
          if (0 == i) {
            data.byteLength = data.byteStrides[i];
          } else {
            data.byteStrides[i + 1] = data.byteStrides[i];
          }
        }
        if (data.elemSizes[i] >= 0) {
          if (0 != i) {
            data.elemSizes[i - 1] *= data.elemSizes[i];
          }
        }

        data.elemSizes.erase(data.elemSizes.begin() + i);
        data.elemSubsizes.erase(data.elemSubsizes.begin() + i);
        data.elemStarts.erase(data.elemStarts.begin() + i);
        data.byteStrides.erase(data.byteStrides.begin() + i);
        changed = true;

        LOG_SPEW(data.str());

        break;
      }
    }
  }
}

void subarrays_byte_strides(Type &type) {

  for (Type &child : type.children()) {
    subarrays_byte_strides(child);
  }

  // I'm not a subarray so I can't merge
  if (!std::holds_alternative<SubarrayData>(type.data)) {
    LOG_SPEW("merge_subsize_one: not subarray");
    return;
  }
  SubarrayData &data = std::get<SubarrayData>(type.data);

  for (int i = 0; i < data.ndims(); ++i) {
    if (data.byteStrides[i] < 0) {
      if (0 == i) {
        data.byteStrides[i] = data.byteLength;
      } else {
        data.byteStrides[i] = data.byteStrides[i - 1] * data.elemSizes[i - 1];
      }
    }
  }

  LOG_SPEW(data.str());
}

/* tries to convert as much of the type to subarrays as possible
 */
Type simplify(const Type &type) {

  LOG_SPEW("simplify type. height=" << type.height());

  Type simp = type;
  LOG_SPEW("simplify pass: convert_nodes_to_subarray");
  convert_nodes_to_subarray(simp);
  LOG_SPEW("simplify pass: fuse_subarrays");
  fuse_subarrays(simp);
  subarrays_merge_subsize_one_dims(simp);
  subarrays_byte_strides(simp);

  return simp;
}

/* tries to create an optimal packer for a Type
  returns falsy if unable to
*/
std::shared_ptr<Packer> plan_pack(Type &type) {

  if (!type) {
    LOG_WARN("couldn't optimize packing strategy for unknown type");
    return nullptr;
  }

  Type simp = simplify(type);

  if (simp.height() == 1 && std::holds_alternative<SubarrayData>(simp.data)) {
    // return a subarray packer
    assert(0 && "subarray packer unimplemented");
  }

#if 0
  // propagate stride through the upper levels
  // block length is only relevant at the lowest level
  for (int64_t i = type.num_levels() - 2; i >= 0; --i) {
    Vector &vec = type.levels()[i];
    Vector &child = type.levels()[i + 1];
    if (vec.blockStride < 0) {
      assert(child.blockStride >= 0);
      vec.blockStride = vec.elemStride * child.count * child.blockStride;
      vec.elemStride = -1; // clear now that we have stride in bytes
      LOG_SPEW("assigned level " << i << " blockStride=" << vec.blockStride);
    }
  }


  /* A vector may describe a contiguous sequence of blocks
  if so, we can describe the parent vector in terms of those blocks natively

  */
  {
    bool changed = true;
    while (changed) {
      int levelToErase = -1;
      // look for a level matching the criteria
      for (int64_t i = type.num_levels() - 1; i >= 1; --i) {
        Vector &vec = type.levels()[i];
        Vector &parent = type.levels()[i - 1];
        if (vec.is_contiguous()) {
          assert(parent.blockLength < 0);
          parent.blockLength = vec.count * vec.blockLength * parent.elemLength;
          // clear this since we have the length in bytes now
          parent.elemLength = -1;
          levelToErase = i;
          LOG_DEBUG("merge contiguous " << i << " into " << i - 1);
          break;
        }
      }
      if (levelToErase >= 0) {
        LOG_DEBUG("erase contiguous " << levelToErase);
        type.levels().erase(type.levels().begin() + levelToErase);
        changed = true;
      } else {
        changed = false;
      }
    }
  }


  /* A vector may have the same stride as its parent.
     If the vector is only one block, the parent can natively express the size
     in terms of bytes instead of child elements
  */
  {
    bool changed = true;
    while (changed) {
      int levelToErase = -1;
      for (int64_t i = type.num_levels() - 1; i >= 1; --i) {
        Vector &vec = type.levels()[i];
        Vector &parent = type.levels()[i - 1];
        if (vec.blockStride == parent.blockStride && 1 == vec.count) {
          assert(parent.blockLength < 0);
          assert(parent.elemLength >= 0);
          parent.blockLength = vec.blockLength;
          parent.elemLength = -1;
          levelToErase = i;
          LOG_DEBUG("merge contiguous " << i << " into " << i - 1);
          break;
        }
      }
      if (levelToErase >= 0) {
        LOG_DEBUG("erase contiguous " << levelToErase);
        type.levels().erase(type.levels().begin() + levelToErase);
        changed = true;
      } else {
        changed = false;
      }
    }
  }


  if (type.num_levels() == 2) {
    /* tree is reversed, level 1 is inner and level 2 is outer
     */
    std::shared_ptr<Packer> pPacker = std::make_shared<PackerStride2>(
        type.levels()[1].blockLength, type.levels()[1].count,
        type.levels()[1].blockStride, type.levels()[0].count,
        type.levels()[0].blockStride);
    LOG_DEBUG("selected PackerStride2");
    return pPacker;
  } else if (type.num_levels() == 1) {
    std::shared_ptr<Packer> pPacker = std::make_shared<PackerStride1>(
        type.levels()[0].blockLength, type.levels()[0].count,
        type.levels()[0].blockStride);
    LOG_DEBUG("selected PackerStride1");
    return pPacker;
  } else {
    LOG_WARN("no optimization for type");
    return nullptr;
  }
#endif

  return nullptr;
}