#pragma once

#define REQUIRE(expr)                                                          \
  if (!(expr))                                                                 \
    return false;
