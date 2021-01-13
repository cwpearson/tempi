//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "method.hpp"

/*  run a variety of different random communication pattern benchmarks on the
 * given method
 */
void random(BM::Method *method, const std::vector<int64_t> &scales,
            const std::vector<float> densities, const int nIters);
const char *random_csv_header();