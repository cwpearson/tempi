#include "types.hpp"

#include "logging.hpp"
#include "packer_stride_1.hpp"
#include "packer_stride_2.hpp"

#include <cassert>

/* generally a vector is the same as a 2D subarray starting at offset 0,
   where the first dimension is the stride and the second dimension is the
   count

   There are certain cases where the vector may be a 1D array.
   We rely on a transformation later to handle these.
   if the count is 1, it's just a 1D subarray of size elemStride and subsize
   elemLength
   if the stride is the same as the blocklength, it's contiguous, so it's
   also a 1D subarray with size of elemStride * count and subsize of
   elemLength*count
   so, both of those cases are equivalent.

   the type may be a basic type with a known byte size (or equivalent), so
   handle that too.

*/

std::map<MPI_Datatype, Type> traverseCache;
/*extern*/ std::map<MPI_Datatype, std::shared_ptr<Packer>> packerCache;

// decode an MPI datatyp
// https://www.mpi-forum.org/docs/mpi-2.0/mpi-20-html/node161.htm
// MPI_Type_get_envelope()
// MPI_Type_get_contents()
Type Type::from_mpi_datatype(MPI_Datatype datatype) {

  LOG_SPEW("from_mpi_datatype ");

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
    MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype
    old_type,MPI_Datatype *newtype)
    */
    assert(integers.size() == 3);
    assert(datatypes.size() == 1);

    MPI_Aint lb, extent;
    MPI_Type_get_extent(datatype, &lb, &extent);
    int size;
    MPI_Type_size(datatype, &size);

    // can't tell length of array element from this alone
    VectorData data;
    data.size = size;
    data.extent = extent;
    data.count = integers[0];
    data.blockLength = integers[1];
    data.stride = integers[2];

    // stride is given in elements, so convert to bytes
    {
      MPI_Aint ext, _;
      MPI_Type_get_extent(datatypes[0], &_, &ext);
      data.stride *= ext;
    }

    LOG_SPEW("vector ->" << data.str());
    ret.data = data;
    Type child = Type::from_mpi_datatype(datatypes[0]);
    ret.children_.push_back(child);
    return ret;

  } else if (MPI_COMBINER_HVECTOR == combiner) {
    Type ret;
    MPI_Type_get_contents(datatype, integers.size(), addresses.size(),
                          datatypes.size(), integers.data(), addresses.data(),
                          datatypes.data());

    /*
    MPI_Type_create_hvector(count, blocklength, stride, oldtype, &newtype);
    */
    assert(integers.size() == 2);
    assert(addresses.size() == 1);
    assert(datatypes.size() == 1);

    MPI_Aint lb, extent;
    MPI_Type_get_extent(datatype, &lb, &extent);
    int size;
    MPI_Type_size(datatype, &size);

    // can't tell length of array element from this alone
    VectorData data;
    data.size = size;
    data.extent = extent;
    data.blockLength = integers[1];
    data.stride = addresses[0];
    data.count = integers[0];

    LOG_SPEW("hvector -> " << data.str());
    ret.data = data;
    Type child = Type::from_mpi_datatype(datatypes[0]);
    ret.children_.push_back(child);
    return ret;
  } else if (MPI_COMBINER_NAMED == combiner) {
    // represent nameed types as a 1D subarray
    LOG_SPEW("named type");
    Type ret;
    DenseData data;

    MPI_Aint lb, extent;
    MPI_Type_get_extent(datatype, &lb, &extent);
    data.extent = extent;

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
    LOG_SPEW("subarray");

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
      data.elemSizes.push_back(integers[1 + ndims * 0 + i]);
      data.elemSubsizes.push_back(integers[1 + ndims * 1 + i]);
      data.elemStarts.push_back(integers[1 + ndims * 2 + i]);
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
    LOG_SPEW("miss " << uintptr_t(datatype) << " in traverse cache");
    Type result = Type::from_mpi_datatype(datatype);
    if (Type() != result) {
      traverseCache[datatype] = result;
    }
    return result;
  }
};

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
    // LOG_SPEW("no fuse: not subarray");
    return;
  }

  // if more than one child, can't fuse
  if (type.children().size() != 1) {
    // LOG_SPEW("no fuse: need 1 child (have " << type.children().size() <<
    // ")");
    return;
  }

  Type &child = type.children()[0];

  // my child is not a subarray so I can't fuse with it
  if (!std::holds_alternative<SubarrayData>(child.data)) {
    // LOG_SPEW("no fuse: child not subarray");
    return;
  }

  // at this point, both me and my single child are subarrays
  // nest the child subarray inside the parent subarray
  SubarrayData fused;
  // first dimensions are child data, then parent data
  const SubarrayData &pData = std::get<SubarrayData>(type.data);
  const SubarrayData &cData = std::get<SubarrayData>(child.data);

  // LOG_SPEW("fuse: " << cData.str() << " + " << pData.str());

  for (size_t dim = 0; dim < cData.ndims(); ++dim) {
    fused.elemStarts.push_back(cData.elemStarts[dim]);
    fused.elemSubsizes.push_back(cData.elemSubsizes[dim]);
    fused.elemSizes.push_back(cData.elemSizes[dim]);
  }
  for (size_t dim = 0; dim < pData.ndims(); ++dim) {
    fused.elemStarts.push_back(pData.elemStarts[dim]);
    fused.elemSubsizes.push_back(pData.elemSubsizes[dim]);
    fused.elemSizes.push_back(pData.elemSizes[dim]);
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

/* Fold two vectors into the parent

   if the child is only one block of data, each parent block actually is a
   child block. the parent may have more than one block (with a correspondingly
   larger stride) so multiply the child blocklength into the parent's

   If the count=1, the nthe vector is one block

   The vector may also be made up of contiguous child types.
   We can't generally detect this because the stride is expressed in bytes
   Right now, we detect a version of this case where the size=extent
   TODO: we either need to store the element-stride so we can see if the
   block length matches the element stride, or we need to store the
   child type extent so we can see if the stride matches the child extent.
   Alternatively, we may be able to make a pass where we try to expand a dense
   type upwards into whatever is above it.

 */
void fold_vectors(Type &type) {

  // try to fold all children into their parents
  for (Type &child : type.children()) {
    fold_vectors(child);
  }

  // no fuse if I'm not a vector
  if (!std::holds_alternative<VectorData>(type.data)) {
    return;
  }

  if (type.children().size() != 1) {
    return;
  }
  Type &child = type.children()[0];

  // my child is not a subarray so I can't fuse with it
  if (!std::holds_alternative<VectorData>(child.data)) {
    return;
  }

  VectorData &pData = std::get<VectorData>(type.data);
  const VectorData &cData = std::get<VectorData>(child.data);

  bool oneBlock = (cData.count == 1);
  oneBlock |= (cData.size == cData.extent);

  if (oneBlock) {
    LOG_SPEW("folding " << cData.str() << " into " << pData.str());

    assert(cData.stride <= pData.stride * pData.blockLength);
    // blockLength * count is the number of contiguous elements
    pData.blockLength *= cData.blockLength * cData.count;

    // delete the child (only 1 child) and replace with granchildren
    std::vector<Type> gchildren = child.children();
    type.children() = gchildren;

    LOG_SPEW("folded into " << pData.str());
  }
}

void subarrays_merge_subsize_one_dims(Type &type) {
  // try to merge all children (start at the bottom of the tree)
  for (Type &child : type.children()) {
    subarrays_merge_subsize_one_dims(child);
  }

  // I'm not a subarray so I can't merge
  if (!std::holds_alternative<SubarrayData>(type.data)) {
    return;
  }

  SubarrayData &data = std::get<SubarrayData>(type.data);

  /* if a dimension has subSize = 1, that means that it represents only a
     single instance of the lower dimension.
     However, it still spaces out instances of the lower dimension

     if we know the elemSize of the degenerate dimension, we say the lower
     dimension's elemSize is just that much larger
  */

  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < data.ndims(); ++i) {
      if (1 == data.elemSubsizes[i]) {
        LOG_SPEW("remove degenerate dim " << i);
        data.elemSizes[i - 1] *= data.elemSizes[i];
        data.erase_dim(i);
        LOG_SPEW(data.str());
        changed = true;
        break;
      }
    }
  }
  LOG_SPEW("type.height=" << type.height());
}

/* tries to convert as much of the type to subarrays as possible
 */
Type simplify(const Type &type) {

  LOG_SPEW("simplify type. height=" << type.height());

  Type simp = type;
  LOG_SPEW("simplify pass: fuse_subarrays");
  fuse_subarrays(simp);
  LOG_SPEW("simplify pass: subarrays_merge_subsize_one_dims");
  subarrays_merge_subsize_one_dims(simp);
  LOG_SPEW("simplify pass: fold_vectors");
  fold_vectors(simp);
  LOG_SPEW("simplify done");
  LOG_SPEW("type.height=" << simp.height());
  return simp;
}

/* tries to create an optimal packer for a Type
  returns falsy if unable to
*/
std::shared_ptr<Packer> plan_pack(Type &type) {

  if (!type) {
    LOG_WARN("couldn't plan_pack strategy for unknown type");
    return nullptr;
  }

  Type simp = simplify(type);
  LOG_SPEW("type.height=" << simp.height());
  StridedBlock strided = to_strided_block(simp);

  if (strided != StridedBlock()) {
    assert(strided.starts.size() == strided.counts.size());
    assert(strided.starts.size() == strided.strides.size());
    if (1 == strided.starts.size()) {
      std::shared_ptr<Packer> packer = std::make_shared<PackerStride1>(
          strided.blockLength, strided.counts[0], strided.strides[0]);
      return packer;
    } else if (2 == strided.starts.size()) {
      std::shared_ptr<Packer> packer = std::make_shared<PackerStride2>(
          strided.blockLength, strided.counts[0], strided.strides[0],
          strided.counts[1], strided.strides[1]);
      return packer;
    } else {
      // generic subarray packer unimplemented
      return nullptr;
    }
  }

// old optimization code
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

/* try to convert a type into a strided block

  this only works if each node has one child, and each node
  is either a vector or subarray, with the node furthest from the root a dense
  type

*/
StridedBlock to_strided_block(const Type &type) {

  StridedBlock ret;

  std::vector<TypeData> data;

  if (Type() == type) {
    LOG_SPEW("empty type");
    return ret;
  }

  LOG_SPEW("height = " << type.height());

  const Type *cur = &type;
  while (true) {
    LOG_SPEW("add type " << cur->data.index());
    data.push_back(cur->data);
    if (cur->children().size() == 1) {
      cur = &(cur->children()[0]);
      continue;
    } else if (cur->children().empty()) {
      // done descending
      break;
    } else {
      LOG_SPEW("too many children");
      // too many children
      return StridedBlock();
    }
  }

  if (data.empty()) {
    LOG_SPEW("no children");
    return ret;
  }

#if TEMPI_OUTPUT_LEVEL >= 4
  for (auto &d : data) {
    LOG_SPEW(d.index());
  }
#endif

  // deepest child must be DenseData
  if (DenseData *dd = std::get_if<DenseData>(&data.back())) {
    ret.blockLength = dd->extent;
    LOG_SPEW("filled blockLength from DenseData: " << ret.str());
  } else {
    LOG_SPEW("type is not built on dense");
    return ret;
  }

  // start from the second-deepest child
  for (int64_t i = data.size() - 2; i >= 0; --i) {
    if (std::holds_alternative<SubarrayData>(data[i])) {
      SubarrayData td = std::get<SubarrayData>(data[i]);
      LOG_SPEW(td.str());

      for (size_t d = 0; d < td.ndims() - 1; ++d) {
        if (0 == ret.ndims()) {
          ret.blockLength *= td.elemSubsizes[d];
          ret.add_dim(td.elemStarts[d], td.elemSubsizes[d + 1],
                      td.elemSizes[d]);
        } else {
          ret.add_dim(td.elemStarts[d], td.elemSubsizes[d + 1],
                      ret.strides[ret.ndims() - 1] * td.elemSizes[d]);
        }
      }

      LOG_SPEW(ret.str());

    } else if (std::holds_alternative<VectorData>(data[i])) {
      VectorData td = std::get<VectorData>(data[i]);
      LOG_SPEW(td.str());

      if (0 == ret.ndims()) {
        ret.blockLength *= td.blockLength;
        ret.add_dim(0 /*no vec offset*/, td.count, td.stride);
      } else if (td.blockLength == 1) {
        ret.add_dim(0 /*no vec offset*/, td.count, td.stride);
      } else {
        LOG_FATAL("how to handle this case?");
      }

      LOG_SPEW(ret.str());

    } else {
      LOG_SPEW("incompatible type");
      return StridedBlock();
    }
  }
  return ret;
}
