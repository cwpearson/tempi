//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#include "env.hpp"
#include "logging.hpp"
#include "packer.hpp"
#include "sender.hpp"
#include "symbols.hpp"
#include "type_cache.hpp"
#include "types.hpp"

#include <mpi.h>

extern "C" int MPI_Type_commit(PARAMS_MPI_Type_commit) {
  int result = libmpi.MPI_Type_commit(ARGS_MPI_Type_commit);

  if (environment::noTempi) {
    return result;
  }
  if (environment::noTypeCommit) {
    return result;
  }

  if (MPI_SUCCESS != result) {
    LOG_ERROR("error in system MPI_Type_commit call");
    return result;
  }

  /* analyze type if never seen before*/
  if (typeCache.count(*datatype)) {
    LOG_SPEW("found MPI_Datatype " << uintptr_t(*datatype) << " in typeCache");
    return result;
  } else {
    TypeRecord record{};

    Type type = traverse(*datatype);
    type = simplify(type);
    StridedBlock sb = to_strided_block(type);
    record.desc = sb;
    std::unique_ptr<Packer> pPacker = plan_pack(sb);
    if (pPacker) {
      LOG_SPEW("cache packer for datatype=" << uintptr_t(*datatype));
    } else {
      LOG_WARN("packer for " << uintptr_t(*datatype) << " was null");
    }

    record.packer = std::move(pPacker);

    // set a sender and reciever, leave null if can't
    if (1 == sb.ndims()) {
      switch (environment::contiguous) {
      case ContiguousMethod::AUTO: {
        LOG_SPEW("SendRecv1D for datatype=" << uintptr_t(*datatype));
        record.sender = std::make_unique<SendRecv1D>(sb);
        record.recver = std::make_unique<SendRecv1D>(sb);
        break;
      }
      case ContiguousMethod::STAGED: {
        LOG_SPEW("SendRecv1DStaged for datatype=" << uintptr_t(*datatype));
        record.sender = std::make_unique<SendRecv1DStaged>(sb);
        record.recver = std::make_unique<SendRecv1DStaged>(sb);
        break;
      }
      case ContiguousMethod::NONE: {
        LOG_SPEW("null sender/recver for datatype=" << uintptr_t(*datatype));
        record.sender = nullptr;
        record.recver = nullptr;
        break;
      }
      }

    } else if (2 == sb.ndims() || 3 == sb.ndims()) {
      switch (environment::datatype) {
      case DatatypeMethod::AUTO: {
        LOG_SPEW("SendRecvND for datatype=" << uintptr_t(*datatype));
        record.sender = std::make_unique<SendRecvND>(sb);
        record.recver = std::make_unique<SendRecvND>(sb);
        break;
      }
      case DatatypeMethod::ONESHOT: {
        LOG_SPEW("OneshotND for datatype=" << uintptr_t(*datatype));
        record.sender = std::make_unique<OneshotND>(sb);
        record.recver = std::make_unique<OneshotND>(sb);
        break;
      }
      case DatatypeMethod::DEVICE: {
        LOG_SPEW("DeviceND for datatype=" << uintptr_t(*datatype));
        record.sender = std::make_unique<DeviceND>(sb);
        record.recver = std::make_unique<DeviceND>(sb);
        break;
      }
      case DatatypeMethod::STAGED: {
        LOG_SPEW("StagedND for datatype=" << uintptr_t(*datatype));
        record.sender = std::make_unique<StagedND>(sb);
        record.recver = std::make_unique<StagedND>(sb);
        break;
      }
      default: {
        LOG_FATAL("unexpected DatatypeMethod");
      }
      }
    } else {
      LOG_WARN("NULL sender for " << uintptr_t(*datatype));
      record.sender = nullptr;
      record.recver = nullptr;
    }

    typeCache[*datatype] = std::move(record);
  }

  return result;
}