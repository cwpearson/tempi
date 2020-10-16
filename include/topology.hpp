#pragma once

#include <mpi.h>

#include <vector>


extern std::vector<int> colocatedRanks;

void topology_init();

// true if this rank is colocated with other
bool is_colocated(int other);
