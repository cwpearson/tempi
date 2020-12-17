#pragma once

#include <cstdint>
// #include <iostream>

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

// largest power of 2 <= x
inline uint8_t log2_floor(int64_t x) {
  if (0 == x) {
    return 0;
  }
  return 63 - __builtin_clzll(x);
}

// smallest power of 2 >= x
inline uint8_t log2_ceil(int64_t x) {
  uint8_t l2 = log2_floor(x);
  // if x is not a power of 2, add 1
  if (x & (x - 1)) {
    ++l2;
  }
  return l2;
}