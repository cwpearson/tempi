#include "types.hpp"

#include "logging.hpp"
#include "packer_stride_1.hpp"
#include "packer_stride_2.hpp"

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
      // vector's size is in terms of the child datatype
      vec.elemLength = integers[1];
      vec.elemStride = integers[2];

      LOG_DEBUG("vector count=" << vec.count
                                << " blockLength=" << vec.blockLength
                                << " blockStride=" << vec.blockStride
                                << " elemLength=" << vec.elemLength
                                << " elemStride=" << vec.elemStride);

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
      // hvector provides an actual byte stride
      vec.elemLength = integers[1];
      vec.blockStride = addresses[0];
      LOG_DEBUG("hvector count=" << vec.count
                                 << " blockLength=" << vec.blockLength
                                 << " byteStride=" << vec.blockStride
                                 << " elemLength=" << vec.elemLength
                                 << " elemStride=" << vec.elemStride);
      type.levels().push_back(vec);
      LOG_DEBUG("added level " << type.num_levels() - 1);
      traverse_helper(type, datatypes[0]);
    } else if (MPI_COMBINER_NAMED == combiner) {
      /* we know the actual size of named types
       */
      LOG_DEBUG("named type");
      if (MPI_BYTE == datatype) {
        Vector vec;
        vec.count = 1;
        vec.blockLength = 1;
        vec.blockStride = 1;
        vec.elemStride = 1;
        vec.elemLength = 1;
        type.levels().push_back(vec);
        LOG_DEBUG("added level " << type.num_levels() - 1);
      } else {
        LOG_FATAL("unhandled named type");
      }
    } else if (MPI_COMBINER_INDEXED_BLOCK == combiner) {
      LOG_DEBUG("indexed_block");
      LOG_WARN("couldn't convert indexed block to structured type");
      type = Type::unknown();
    } else if (MPI_COMBINER_HINDEXED_BLOCK == combiner) {
      LOG_DEBUG("hindexed_block");
      LOG_WARN("couldn't convert hindexed block to structured type");
      type = Type::unknown();
      return;
    } else if (MPI_COMBINER_HINDEXED == combiner) {
      // http://www.cse.chalmers.se/~mckee/papers/sc03.pdf
      LOG_DEBUG("hindexed");
      LOG_WARN("couldn't convert hindexed to structured type");
      type = Type::unknown();
      return;
    } else if (MPI_COMBINER_CONTIGUOUS == combiner) {
      LOG_DEBUG("contiguous");
      LOG_WARN("couldn't convert contiguous to structured type");
      type = Type::unknown();
      return;
    } else if (MPI_COMBINER_STRUCT == combiner) {
      LOG_DEBUG("struct");
      LOG_WARN("couldn't convert struct to structured type");
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
    LOG_SPEW("miss " << uintptr_t(datatype) << " in traverse cache.");
    Type result;
    traverse_helper(result, datatype);
    if (Type::unknown() != result) {
      traverseCache[datatype] = result;
    }
    return result;
  }
};

/* tries to create an optimal packer for a Type
  returns falsy if unable to
*/
std::shared_ptr<Packer> plan_pack(Type &type) {

  if (Type::unknown() == type) {
    LOG_WARN("couldn't plan packing strategy for unknown type");
    return nullptr;
  }

  LOG_SPEW("PRINT TREE");
  for (int64_t i = type.num_levels() - 1; i >= 0; --i) {
#if TEMPI_OUTPUT_LEVEL >= 5
    Vector &vec = type.levels()[i];
#endif
    LOG_SPEW("level " << i << ": bstride=" << vec.blockStride << " blength="
                      << vec.blockLength << " estride=" << vec.elemStride
                      << " elength=" << vec.elemLength << " cnt=" << vec.count);
  }

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

  LOG_SPEW("PRINT TREE");
  for (int64_t i = type.num_levels() - 1; i >= 0; --i) {
#if TEMPI_OUTPUT_LEVEL >= 5
    Vector &vec = type.levels()[i];
#endif
    LOG_SPEW("level " << i << ": bstride=" << vec.blockStride << " blength="
                      << vec.blockLength << " estride=" << vec.elemStride
                      << " elength=" << vec.elemLength << " cnt=" << vec.count);
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

  LOG_SPEW("PRINT TREE");
  for (int64_t i = type.num_levels() - 1; i >= 0; --i) {
#if TEMPI_OUTPUT_LEVEL >= 5
    Vector &vec = type.levels()[i];
#endif
    LOG_SPEW("level " << i << ": bstride=" << vec.blockStride << " blength="
                      << vec.blockLength << " estride=" << vec.elemStride
                      << " elength=" << vec.elemLength << " cnt=" << vec.count);
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

  LOG_SPEW("PRINT TREE");
  for (int64_t i = type.num_levels() - 1; i >= 0; --i) {
#if TEMPI_OUTPUT_LEVEL >= 5
    Vector &vec = type.levels()[i];
#endif
    LOG_SPEW("level " << i << ": bstride=" << vec.blockStride << " blength="
                      << vec.blockLength << " estride=" << vec.elemStride
                      << " elength=" << vec.elemLength << " cnt=" << vec.count);
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
}