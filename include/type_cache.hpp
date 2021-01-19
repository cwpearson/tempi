//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)



/* kept out of types.cpp so that benchmark code can use the typeCache directly
 * without bring in c++17 for std::variant in TypeData
 */

#pragma once

#include "packer.hpp"
#include "sender.hpp"
#include "strided_block.hpp"

#include <mpi.h>

#include <unordered_map>
#include <memory>

struct TypeRecord {
  std::unique_ptr<Packer> packer;
  StridedBlock desc;
  std::unique_ptr<Sender> sender;
  std::unique_ptr<Recver> recver;
};

extern std::unordered_map<MPI_Datatype, TypeRecord> typeCache;