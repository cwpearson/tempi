#pragma once

#include <cstdint>

inline int64_t next_power_of_two(int64_t x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  x++;
  return x;
}

inline int32_t next_power_of_two(int32_t x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return x;
}

inline int64_t div_ceil(int64_t n, int64_t d) { return (n + d - 1) / d; }