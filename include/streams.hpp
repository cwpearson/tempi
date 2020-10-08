#pragma once

#include <cuda_runtime.h>

#include <vector>

extern std::vector<cudaStream_t> commStream;
extern std::vector<cudaStream_t> kernStream;

void streams_init();
void streams_finalize();