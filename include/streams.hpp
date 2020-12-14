#pragma once

#include <cuda_runtime.h>

extern cudaStream_t commStream;
extern cudaStream_t kernStream;

void streams_init();
void streams_finalize();