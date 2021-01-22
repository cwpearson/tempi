//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "symbols.hpp"

namespace neighbor_alltoallw {

/*use Isend/Irecv*/
int isir(PARAMS_MPI_Neighbor_alltoallw);

} // namespace neighbor_alltoallw

