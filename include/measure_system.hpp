#pragma once

#include <mpi.h>

struct KernelLaunch {
  double secs;
};

extern KernelLaunch kernelLaunch;

void measure_system(MPI_Comm comm);