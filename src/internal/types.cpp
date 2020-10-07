#include "types.hpp"

#include "logging.hpp"

#include <cassert>

std::map<MPI_Datatype, Type> traverseCache;
/*extern*/ std::map<MPI_Datatype, std::shared_ptr<Packer>> packerCache;

void traverse_helper(Type &type, MPI_Datatype datatype) {
  // decode a possible strided type
  // https://www.mpi-forum.org/docs/mpi-2.0/mpi-20-html/node161.htm
  // MPI_Type_get_envelope()
  // MPI_Type_get_contents()
  // MPI_Type_vector(int count, int blocklength, int stride,MPI_Datatype
  // old_type,MPI_Datatype *newtype_p)
  {
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

      MPI_Type_get_contents(datatype, integers.size(), addresses.size(),
                            datatypes.size(), integers.data(), addresses.data(),
                            datatypes.data());

      /*
      MPI_Type_vector(int count, int blocklength, int stride,MPI_Datatype
      old_type,MPI_Datatype *newtype_p)
      */
      assert(integers.size() == 3);
      assert(datatypes.size() == 1);

      Vector vec;
      vec.count = integers[0];
      vec.blockLength = integers[1];
      vec.elemStride = integers[2];

      LOG_DEBUG("vector count=" << vec.count
                                << " blockLength=" << vec.blockLength
                                << " elemStride=" << vec.elemStride
                                << " byteStride=" << vec.byteStride);

      type.levels().push_back(vec);
      LOG_DEBUG("added level " << type.num_levels() - 1);

      traverse_helper(type, datatypes[0]);

    } else if (MPI_COMBINER_HVECTOR == combiner) {

      MPI_Type_get_contents(datatype, integers.size(), addresses.size(),
                            datatypes.size(), integers.data(), addresses.data(),
                            datatypes.data());

      /*
      MPI_Type_create_hvector(count, blocklength, stride, planeType, &fullType);
      */
      assert(integers.size() == 2);
      assert(addresses.size() == 1);
      assert(datatypes.size() == 1);
      Vector vec;
      vec.count = integers[0];
      vec.blockLength = integers[1];
      vec.byteStride = addresses[0];
      LOG_DEBUG("hvector count=" << vec.count
                                 << " blockLength=" << vec.blockLength
                                 << " byteStride=" << vec.byteStride);
      type.levels().push_back(vec);
      LOG_DEBUG("added level " << type.num_levels() - 1);
      traverse_helper(type, datatypes[0]);
    } else if (MPI_COMBINER_NAMED == combiner) {
      LOG_DEBUG("named type");
      if (MPI_BYTE == datatype) {
        Vector vec;
        vec.count = 1;
        vec.blockLength = 1;
        vec.byteStride = 1;
        type.levels().push_back(vec);
        LOG_DEBUG("added level " << type.num_levels() - 1);
      } else {
        LOG_FATAL("unhandled named type");
      }
    } else if (MPI_COMBINER_INDEXED_BLOCK == combiner) {
      LOG_DEBUG("indexed_block");
      LOG_WARN("couldn't convert to structured type");
      type = Type::unknown();
    } else if (MPI_COMBINER_HINDEXED_BLOCK == combiner) {
      LOG_DEBUG("hindexed_block");
      LOG_WARN("couldn't convert to structured type");
      type = Type::unknown();
      return;
    } else if (MPI_COMBINER_HINDEXED == combiner) {
      LOG_DEBUG("hindexed");
      LOG_WARN("couldn't convert to structured type");
      type = Type::unknown();
      return;
    } else if (MPI_COMBINER_CONTIGUOUS == combiner) {
      LOG_DEBUG("contiguous");
      LOG_WARN("couldn't convert to structured type");
      type = Type::unknown();
      return;
    } else if (MPI_COMBINER_STRUCT == combiner) {
      LOG_DEBUG("struct");
      LOG_WARN("couldn't convert to structured type");
      type = Type::unknown();
      return;
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
    MPI_COMBINER_SUBARRAY MPI_TYPE_CREATE_SUBARRAY
    MPI_COMBINER_DARRAY MPI_TYPE_CREATE_DARRAY
    MPI_COMBINER_F90_REAL MPI_TYPE_CREATE_F90_REAL
    MPI_COMBINER_F90_COMPLEX MPI_TYPE_CREATE_F90_COMPLEX
    MPI_COMBINER_F90_INTEGER MPI_TYPE_CREATE_F90_INTEGER
    MPI_COMBINER_RESIZED MPI_TYPE_CREATE_RESIZED


    If combiner is MPI_COMBINER_NAMED then datatype is a named predefined
    datatype.
    */
  }
}

Type traverse(MPI_Datatype datatype) {
  if (0 != traverseCache.count(datatype)) {
    return traverseCache[datatype];
  } else {
    Type result;
    traverse_helper(result, datatype);
    if (Type::unknown() != result) {
      traverseCache[datatype] = result;
    }
    return result;
  }
};

std::shared_ptr<Packer> plan_pack(Type &type) {

  if (Type::unknown() == type) {
    LOG_WARN("couldn't making packing strategy");
    return nullptr;
  }

  LOG_SPEW("PRINT TREE");
  for (int64_t i = type.num_levels() - 1; i >= 0; --i) {
    Vector &vec = type.levels()[i];
    LOG_SPEW("level " << i << ": bstride=" << vec.byteStride << " blength="
                      << vec.blockLength << " cnt=" << vec.count);
  }

  // use propogate strides up from the bottom of the tree
  for (int64_t i = type.num_levels() - 2; i >= 0; --i) {
    Vector &vec = type.levels()[i];
    if (vec.byteStride < 0) {
      Vector &child = type.levels()[i + 1];
      assert(child.byteStride >= 0);
      vec.byteStride = child.count * child.byteStride * vec.elemStride;
      LOG_SPEW("assigned level " << i << " byteStride " << vec.byteStride);
    }
  }

  LOG_SPEW("PRINT TREE");
  for (int64_t i = type.num_levels() - 1; i >= 0; --i) {
    Vector &vec = type.levels()[i];
    LOG_SPEW("level " << i << ": bstride=" << vec.byteStride << " blength="
                      << vec.blockLength << " cnt=" << vec.count);
  }

  /* a level may describe a contiguous block (this includes primitive types)
  if so, multiply it's count * blockLength into the parent's blocklength and
  remove it
  */
  {
    bool changed = true;
    while (changed) {
      int levelToErase = -1;
      // look for a level matching the criteria
      for (int64_t i = type.num_levels() - 1; i >= 1; --i) {
        Vector &vec = type.levels()[i];
        Vector &parent = type.levels()[i - 1];
        if (vec.byteStride == vec.blockLength) {
          parent.blockLength *= vec.count * vec.blockLength;
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

  LOG_SPEW("PRINT TREE");
  for (int64_t i = type.num_levels() - 1; i >= 0; --i) {
    Vector &vec = type.levels()[i];
    LOG_SPEW("level " << i << ": bstride=" << vec.byteStride << " blength="
                      << vec.blockLength << " cnt=" << vec.count);
  }

  if (type.num_levels() == 2) {
    /* tree is reversed, level 1 is inner and level 2 is outer
     */
    std::shared_ptr<Packer> pPacker = std::make_shared<PackerStride2>(
        type.levels()[1].blockLength, type.levels()[1].count,
        type.levels()[1].byteStride, type.levels()[0].count,
        type.levels()[0].byteStride);
    return pPacker;
  } else {
    LOG_WARN("no optimization for type");
    return nullptr;
  }
}