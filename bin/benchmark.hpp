#pragma once

#include "method.hpp"

/*  run a variety of different random communication pattern benchmarks on the
 * given method
 */
void random(BM::Method *method, const std::vector<int64_t> &scales,
            const std::vector<float> densities, const int nIters);
const char *random_csv_header();