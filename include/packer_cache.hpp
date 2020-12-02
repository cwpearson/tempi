/* kept out of types.cpp so that benchmark code can use the packerCache directly without bring in c++17 in types.hpp
*/

#pragma once

#include "packer.hpp"

#include <mpi.h>

#include <map>
#include <memory>


extern std::map<MPI_Datatype, std::unique_ptr<Packer>> packerCache;