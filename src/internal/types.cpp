//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "types.hpp"

#include "env.hpp"
#include "logging.hpp"
#include "packer_1d.hpp"
#include "packer_2d.hpp"
#include "packer_3d.hpp"
#include "type_cache.hpp"

#include <algorithm>
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

// decode an MPI datatype
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
    /* a vector is two streams, the child representing the repeated elements in
       a block and the parent representing the repeated blocks
    */

    Type parent, child;

    MPI_Type_get_contents(datatype, integers.size(), addresses.size(),
                          datatypes.size(), integers.data(), addresses.data(),
                          datatypes.data());

    /*
    MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype
    old_type,MPI_Datatype *newtype)
    */
    assert(integers.size() == 3);
    assert(datatypes.size() == 1);
    int count = integers[0];
    int blocklength = integers[1];
    int stride = integers[2];
    MPI_Datatype old_type = datatypes[0];

    MPI_Aint lb, extent, oldLb, oldExtent;
    int size, oldSize;
    MPI_Type_get_extent(datatype, &lb, &extent);
    parent.extent = extent;
    MPI_Type_size(datatype, &size);
    MPI_Type_get_extent(old_type, &oldLb, &oldExtent);
    MPI_Type_size(datatype, &oldSize);

    StreamData cData{};
    cData.off = 0;
    cData.count = blocklength;
    cData.stride = oldExtent;

    StreamData pData{};
    pData.off = 0;
    pData.count = count;
    pData.stride = oldExtent * stride; // size of elements * stride in elements

    parent.data = pData;
    child.data = cData;
    LOG_SPEW("vector pData ->" << pData.str());
    LOG_SPEW("vector cData ->" << cData.str());

    // build the type from old_type
    Type gchild = Type::from_mpi_datatype(datatypes[0]);
    LOG_SPEW("before vector, height=" << gchild.height());
    child.children_.push_back(gchild);

    // add child to parent
    parent.children_.push_back(child);

    LOG_SPEW("after vector, height=" << parent.height());

    return parent;

  } else if (MPI_COMBINER_HVECTOR == combiner) {
    /* a vector is two streams, the child representing the repeated elements in
      a block and the parent representing the repeated blocks
   */

    Type parent, child;

    MPI_Type_get_contents(datatype, integers.size(), addresses.size(),
                          datatypes.size(), integers.data(), addresses.data(),
                          datatypes.data());

    /*
    MPI_Type_vector(int count, int blocklength, MPI_Aint stride, MPI_Datatype
    oldtype,MPI_Datatype *newtype)
    */
    assert(integers.size() == 2);
    assert(addresses.size() == 1);
    assert(datatypes.size() == 1);
    int count = integers[0];
    int blocklength = integers[1];
    int stride = addresses[0];
    MPI_Datatype oldtype = datatypes[0];

    MPI_Aint lb, extent, oldLb, oldExtent;
    int size, oldSize;
    MPI_Type_get_extent(datatype, &lb, &extent);
    parent.extent = extent;
    MPI_Type_size(datatype, &size);
    MPI_Type_get_extent(oldtype, &oldLb, &oldExtent);
    MPI_Type_size(datatype, &oldSize);

    StreamData pData{};
    pData.off = 0; // vector has no offset
    pData.count = count;
    pData.stride =
        stride; // give in bytes instead of child elements (as in vector)

    StreamData cData{};
    cData.off = 0; // vector has no offset
    cData.count = blocklength;
    cData.stride = oldExtent;

    parent.data = pData;
    child.data = cData;
    LOG_SPEW("hvector pData ->" << pData.str());
    LOG_SPEW("hvector cData ->" << cData.str());

    // build the type from oldtype
    Type gchild = Type::from_mpi_datatype(datatypes[0]);
    child.children_.push_back(gchild);

    // add child to parent
    parent.children_.push_back(child);

    return parent;
  } else if (MPI_COMBINER_NAMED == combiner) {
    LOG_SPEW("named type");
    Type ret;
    DenseData data{};
    data.off = 0;

    MPI_Aint lb, extent;
    MPI_Type_get_extent(datatype, &lb, &extent);
    data.extent = extent;
    ret.extent = extent;

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
    Type ret;
    MPI_Type_get_contents(datatype, integers.size(), addresses.size(),
                          datatypes.size(), integers.data(), addresses.data(),
                          datatypes.data());

    /*
    MPI_Type_contiguous(count, oldtype, &newtype);
    */
    assert(integers.size() == 1);
    assert(datatypes.size() == 1);
    int count = integers[0];
    MPI_Datatype oldtype = datatypes[0];

    {
      MPI_Aint _, e;
      MPI_Type_get_extent(datatype, &_, &e);
      ret.extent = e;
    }

    MPI_Aint oldLb, oldExtent;
    int oldSize;
    MPI_Type_get_extent(oldtype, &oldLb, &oldExtent);
    MPI_Type_size(oldtype, &oldSize);

    StreamData data{};
    data.off = 0;
    data.count = count;
    data.stride = oldExtent;

    LOG_SPEW("contiguous -> " << data.str());
    ret.data = data;
    Type child = Type::from_mpi_datatype(oldtype);
    ret.children_.push_back(child);
    return ret;
  } else if (MPI_COMBINER_STRUCT == combiner) {
    LOG_DEBUG("struct");
    LOG_WARN("couldn't convert struct to a Type");
    return Type();
  } else if (MPI_COMBINER_SUBARRAY == combiner) {
    /* ndim subarray is ndim streams. the first dim is the outer dim
     */
    LOG_SPEW("subarray");

    MPI_Type_get_contents(datatype, integers.size(), addresses.size(),
                          datatypes.size(), integers.data(), addresses.data(),
                          datatypes.data());

    std::vector<StreamData> datas;

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
//    if (order != MPI_ORDER_C) {
//     LOG_ERROR("unhandled order in subarray type");
//      return Type();
//   }
    MPI_Datatype oldtype = datatypes[0];

    MPI_Aint oldLb, oldExtent;
    int oldSize;
    MPI_Type_get_extent(oldtype, &oldLb, &oldExtent);
    MPI_Type_size(oldtype, &oldSize);

    for (int i = 0; i < ndims; ++i) {
      // int size = integers[1 + ndims * 0 + i];
      int subsize = integers[1 + ndims * 1 + i]; // subsize[i]
      int start = integers[1 + ndims * 2 + i];

      StreamData data{};
      data.off = start * oldExtent;
      data.stride = oldExtent;
      if (order == MPI_ORDER_C) {
        for (int j = i + 1; j < ndims; ++j) {
          data.stride *= integers[1 + ndims * 0 + j]; // size[j]
          data.off *= integers[1 + ndims * 0 + j];    // size[j]
        } 
      } else { // MPI_ORDER_FORTRAN
        for (int j = i - 1; j >= 0; --j) {
          data.stride *= integers[1 + ndims * 0 + j]; // size[j]
          data.off *= integers[1 + ndims * 0 + j];    // size[j]
        } 
      }
      data.count = subsize;
      datas.push_back(data);
    }

    if (order == MPI_ORDER_C) {
        std::reverse(datas.begin(), datas.end());
    }

    for (int i = 0; i < ndims; ++i) {
      LOG_SPEW("subarray " << i << " -> " << datas[i].str());
    }

    /* build the tree from the bottom up
     */

    Type child = Type::from_mpi_datatype(oldtype);
    LOG_SPEW("below subarray, height=" << child.height());
    for (int i = 0; i < ndims; ++i) {
      Type parent;
      parent.data = datas[i];
      parent.children_.push_back(child);
      child = parent;
    }

    {
      MPI_Aint _, e;
      MPI_Type_get_extent(datatype, &_, &e);
      child.extent = e;
    }

    LOG_SPEW("after subarray, height=" << child.height());
    return child;
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
    LOG_SPEW("found MPI_Datatype " << uintptr_t(datatype)
                                   << " in traverse cache");
    return traverseCache[datatype];
  } else {
    LOG_SPEW("miss " << uintptr_t(datatype) << " in traverse cache");
    Type result = Type::from_mpi_datatype(datatype);
    LOG_SPEW(result.str());

    if (Type() != result) {
      LOG_SPEW("insert MPI_Datatype " << uintptr_t(datatype)
                                      << " into traverse cache");
      traverseCache[datatype] = result;
    }
    return result;
  }
};

/*
  if there are two stream data, ensure the larger extent is higher up
*/
bool stream_swap(Type &type) {

  bool changed = false;

  // only works if I'm a stream and my child is stream
  if (!std::holds_alternative<StreamData>(type.data)) {
    return false;
  }
  assert(1 == type.children().size());
  Type &child = type.children()[0];
  if (!std::holds_alternative<StreamData>(child.data)) {
    return false;
  }

  StreamData &pData = std::get<StreamData>(type.data);
  StreamData &cData = std::get<StreamData>(child.data);

  if (pData.stride < cData.stride) {
    changed = true;
    std::swap(pData, cData);
  }

  // descend into child
  changed |= stream_swap(child);

  return changed;
}

/* if a stream has a dense child, and that stream's stride is the same as the
   child's extent, then the stream is also dense
*/
bool stream_dense_fold(Type &type) {

  bool changed = false;

  // try to fold all children into their parents first
  for (Type &child : type.children()) {
    changed |= stream_dense_fold(child);
  }

  // only works if I'm a stream and my child is dense
  if (!std::holds_alternative<StreamData>(type.data)) {
    return false;
  }
  assert(1 == type.children().size());
  Type &child = type.children()[0];
  if (!std::holds_alternative<DenseData>(child.data)) {
    return false;
  }

  assert(type.data.index() != std::variant_npos);
  assert(child.data.index() != std::variant_npos);

  const StreamData &pData = std::get<StreamData>(type.data);
  DenseData &cData = std::get<DenseData>(child.data);

  if (cData.extent == pData.stride) {
    changed = true;

    // replace this type with it's own child
    Type newType = child;
    DenseData newData = cData;
    newData.extent = pData.count * pData.stride;
    newData.off += pData.off;
    LOG_SPEW("stream_dense_fold: -> " << newData.str());
    newType.data = newData;
    type = newType;
    assert(type.data.index() != std::variant_npos);
  }

  return changed;
}

/* if nested streams, if the child count is 1, the parent's
elements are just the child's element

 */
bool stream_elision(Type &type) {

  bool changed = false;

  // try to fold all children into their parents first
  for (Type &child : type.children()) {
    changed |= stream_elision(child);
  }

  // type and child must be StreamData
  if (!std::holds_alternative<StreamData>(type.data)) {
    return false;
  }
  assert(1 == type.children().size());
  Type &child = type.children()[0];
  if (!std::holds_alternative<StreamData>(child.data)) {
    return false;
  }

  const StreamData &cData = std::get<StreamData>(child.data);

  if (1 == cData.count) {
    changed = true;
    // erase child and replace with gchild
    // delete the child (only 1 child) and replace with granchildren
    std::vector<Type> gchildren = child.children();
    type.children() = gchildren;
  }

  return changed;
}

/* if a stream has count 1, it can be removed

 */
bool stream_elision2(Type &type) {

  bool changed = false;

  // try to remove all size 1 children first
  for (Type &child : type.children()) {
    changed |= stream_elision2(child);
  }

  // type must be StreamData
  if (!std::holds_alternative<StreamData>(type.data)) {
    return false;
  }
  assert(1 == type.children().size());

  const StreamData &tData = std::get<StreamData>(type.data);

  if (1 == tData.count) {
    changed = true;
    // replace this guy with his child
    Type child = type.children()[0];
    child.extent = type.extent;
    type = child;
  }

  return changed;
}

/* detect, for example, where two vectors of two blocks is just one vector of
  four blocks

  when stride of parent is count * stride of child

  parent/child replaced with something like child, except count is now
  child.count * parent.count

  this probably only comes in vectors of subarrays since subarrays have
  padding on the end
*/
bool stream_flatten(Type &type) {

  bool changed = false;

  // try to fold all children into their parents first
  for (Type &child : type.children()) {
    changed |= stream_flatten(child);
  }

  // type and child must be StreamData
  if (!std::holds_alternative<StreamData>(type.data)) {
    return false;
  }
  assert(1 == type.children().size());
  Type &child = type.children()[0];
  if (!std::holds_alternative<StreamData>(child.data)) {
    return false;
  }

  StreamData &pData = std::get<StreamData>(type.data);
  const StreamData &cData = std::get<StreamData>(child.data);

  if (pData.stride == cData.count * cData.stride) {
    changed = true;

    // transform parent in to child, with count multiplied by parents count
    pData.count *= cData.count;
    pData.stride = cData.stride;
    pData.off += cData.off;
    std::vector<Type> gchildren = child.children();
    type.children() = gchildren;
  }

  return changed;
}

/* tries to convert as much of the type to subarrays as possible
 */
Type simplify(const Type &type) {
  LOG_SPEW("simplify()");

  int iter = 0;
  Type simp = type;

  LOG_SPEW("before passes");
  LOG_SPEW("\n" + simp.str());

  bool changed = true;
  while (changed) {
    changed = false;
    ++iter;
    LOG_SPEW("optimization iter " << iter);

    /* new, less-tested swapper with new stream elision
       if problems, remove the swap and go to stream_elision
    */
    changed |= stream_swap(simp);
    LOG_SPEW("after stream_swap");
    LOG_SPEW("\n" + simp.str());
    changed |= stream_dense_fold(simp);
    LOG_SPEW("after stream_dense_fold");
    LOG_SPEW("\n" + simp.str());
    changed |= stream_flatten(simp);
    LOG_SPEW("after stream_flatten");
    LOG_SPEW("\n" + simp.str());
    changed |= stream_elision2(simp);
    LOG_SPEW("after stream_elision2");
    LOG_SPEW("\n" + simp.str());
  }

  /* TODO
     Here, we sort by the stride after doing canonicalization

  */

  changed = true;
  while (changed) {
    changed = false;
    ++iter;
    LOG_SPEW("sort iter " << iter);
  }

  LOG_SPEW("simplify done. " << iter << " iterations");
  LOG_SPEW("type.height=" << simp.height());
  return simp;
}

/* tries to create an optimal packer for a Type
  returns falsy if unable to
*/
std::unique_ptr<Packer> plan_pack(const StridedBlock &sb) {

  if (sb != StridedBlock()) {
    if (1 == sb.ndims()) {
      LOG_SPEW("select Packer1D for " << sb.str());
      std::unique_ptr<Packer> packer =
          std::make_unique<Packer1D>(sb.start_, sb.counts[0]);
      return packer;
    } else if (2 == sb.ndims()) {
      LOG_SPEW("select Packer2D for " << sb.str());
      std::unique_ptr<Packer> packer = std::make_unique<Packer2D>(
          sb.start_, sb.counts[0], sb.counts[1], sb.strides[1], sb.extent_);
      return packer;
    } else if (3 == sb.ndims()) {
      LOG_SPEW("select Packer3D for " << sb.str());
      std::unique_ptr<Packer> packer = std::make_unique<Packer3D>(
          sb.start_, sb.counts[0], sb.counts[1], sb.strides[1], sb.counts[2],
          sb.strides[2], sb.extent_);
      return packer;
    } else {
      LOG_SPEW("no packer for " << sb.str());
      return nullptr;
    }
  } else {
    LOG_WARN("couldn't plan_pack strategy for unknown type");
    return nullptr;
  }
}

/* try to convert a type into a strided block

  this only works if each type is a stream, except the bottom type, which is
  dense

*/
StridedBlock to_strided_block(const Type &type) {

  std::vector<TypeData> data;

  if (Type() == type) {
    LOG_SPEW("can't convert empty type to strided block");
    return StridedBlock();
  }

  LOG_SPEW("height = " << type.height());

  const Type *cur = &type;
  while (true) {
    data.push_back(cur->data);
    if (cur->children().size() == 1) {
      cur = &(cur->children()[0]);
      continue;
    } else if (cur->children().empty()) {
      // done descending
      break;
    } else {
      LOG_DEBUG("to_strided_block(): too many children");
      // too many children
      return StridedBlock();
    }
  }

  if (data.empty()) {
    LOG_SPEW("empty type");
    return StridedBlock();
  }

  StridedBlock ret;
  ret.extent_ = type.extent;
  assert(ret.extent_ > 0);

  // deepest child must be DenseData
  if (DenseData *dd = std::get_if<DenseData>(&data.back())) {
    ret.add_dim(dd->off, dd->extent, 1 /*1B between each element of dd*/);
    LOG_SPEW("StridedBlock from DenseData -> " << ret.str());
  } else {
    LOG_SPEW("type is not built on dense");
    return StridedBlock();
  }

  // start from the second-deepest child
  for (int64_t i = data.size() - 2; i >= 0; --i) {
    if (std::holds_alternative<StreamData>(data[i])) {
      StreamData sd = std::get<StreamData>(data[i]);
      LOG_SPEW(sd.str());

      ret.add_dim(sd.off, sd.count, sd.stride);

      LOG_SPEW("StridedBlock from StreamData -> " << ret.str());

    } else {
      LOG_SPEW("incompatible type");
      return StridedBlock();
    }
  }
  return ret;
}

void release(MPI_Datatype ty) {
  LOG_SPEW("release MPI_Datatype " << uintptr_t(ty));
  traverseCache.erase(ty);
  typeCache.erase(ty);
}

void types_init() {
  for (MPI_Datatype datatype : {MPI_BYTE, MPI_FLOAT, MPI_DOUBLE}) {
    TypeRecord record{};
    Type type = traverse(datatype);
    type = simplify(type);
    StridedBlock sb = to_strided_block(type);
    record.desc = sb;
    // set a sender and reciever, leave null if can't
    if (1 == sb.ndims()) {
      switch (environment::contiguous) {
      case ContiguousMethod::AUTO: {
        LOG_SPEW("SendRecv1D for datatype=" << uintptr_t(datatype));
        record.sender = std::make_unique<SendRecv1D>(sb);
        record.recver = std::make_unique<SendRecv1D>(sb);
        break;
      }
      case ContiguousMethod::STAGED: {
        LOG_SPEW("SendRecv1DStaged for datatype=" << uintptr_t(datatype));
        record.sender = std::make_unique<SendRecv1DStaged>(sb);
        record.recver = std::make_unique<SendRecv1DStaged>(sb);
        break;
      }
      case ContiguousMethod::NONE: {
        LOG_SPEW("NULL sender for datatype=" << uintptr_t(datatype));
        record.sender = nullptr;
        record.recver = nullptr;
        break;
      }
      }
    } else {
      LOG_WARN("NULL sender for " << uintptr_t(datatype));
      record.sender = nullptr;
      record.recver = nullptr;
    }
    typeCache[datatype] = std::move(record);
  }
}
