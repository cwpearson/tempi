#pragma once

#include "../include/logging.hpp"

// clang-format off
#define REQUIRE(expr) { int _line = __LINE__;                                                         \
  if (!(expr)) {                                                               \
    LOG_FATAL("\"" #expr "\" failed");                                                \
  }\
}
