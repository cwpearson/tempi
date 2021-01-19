//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "counters.hpp"
#include "cuda_runtime.hpp"

template <class T> class host_allocator {
public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef T &reference;
  typedef const T &const_reference;
  typedef T value_type;

  host_allocator() {}
  host_allocator(const host_allocator &) {}

  pointer allocate(size_type n, const void * = 0) {
    TEMPI_COUNTER_OP(hostAllocator, NUM_ALLOCS, ++);
    if (!n) {
      return nullptr;
    }
    T *t = new T[n];
    if (!t) {
      LOG_ERROR("failed to allocate " << n << " elems of size " << sizeof(T));
      throw std::bad_alloc();
    }
    cudaError_t err =
        cudaHostRegister(t, n * sizeof(T), cudaHostRegisterMapped);
    if (cudaSuccess != err) {
      delete[] t;
      CUDA_RUNTIME(err);
      throw std::bad_alloc();
    }
    return t;
  }

  void deallocate(void *p, size_type) {
    TEMPI_COUNTER_OP(hostAllocator, NUM_DEALLOCS, ++);
    if (p) {
      cudaError_t err = cudaHostUnregister(p);
      if (cudaSuccess != err) {
        CUDA_RUNTIME(err);
      }
      delete[](T *) p;
    }
  }

  pointer address(reference x) const { return &x; }
  const_pointer address(const_reference x) const { return &x; }
  host_allocator<T> &operator=(const host_allocator &) { return *this; }
  void construct(pointer p, const T &val) { new ((T *)p) T(val); }
  void destroy(pointer p) { p->~T(); }

  size_type max_size() const { return size_t(-1); }

  template <typename U> struct rebind { typedef host_allocator<U> other; };

  template <typename U> host_allocator(const host_allocator<U> &) {}

  template <typename U> host_allocator &operator=(const host_allocator<U> &) {
    return *this;
  }
};
