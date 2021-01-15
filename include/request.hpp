//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

struct Request {

  static uint32_t count;

  union {
    MPI_Request mpi;
    uint8_t bytes[sizeof(MPI_Request)];
  } data;

  // implicit conversion to MPI_Request for convenience
  operator MPI_Request() { return data.mpi; }

  static Request make() {
    // okay for count to overflow
    return Request(count++);
  }

private:
  /* zero data and copy the first n bytes of count into it */
  Request(uint32_t u) : data({}) {
    std::memcpy(data.bytes, &u, std::min(sizeof(MPI_Request), sizeof(u)));
  }
};