//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "../include/logging.hpp"

// clang-format off
#define REQUIRE(expr) { int _line = __LINE__;                                                         \
  if (!(expr)) {                                                               \
    LOG_FATAL("\"" #expr "\" failed");                                                \
  }\
}
