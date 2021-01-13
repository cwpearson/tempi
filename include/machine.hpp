//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <mpi.h>

class Machine
{
private:
    int tagUpperBound_;

public:
    Machine(MPI_Comm comm);

    // the node that `rank` is on
    int node_of_rank(int rank);

    // number of nodes in the machine
    int num_nodes();

    // the largest available tag
    int mpi_tag_ub();
};
