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
