#pragma once

#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_runtime.hpp"
#include "logging.hpp"

/*! \brief a CUDA device allocator

    respects cudaSetDevice
 */
template <class T> class device_allocator {
public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef T &reference;
  typedef const T &const_reference;
  typedef T value_type;

  device_allocator() {}
  device_allocator(const device_allocator &) {}

  pointer allocate(size_type n, const void * = 0) {
    T *t{};
    cudaError_t err = cudaMalloc(&t, n * sizeof(T));
    if (cudaSuccess != err) {
      throw std::bad_alloc();
    }
    return t;
  }

  void deallocate(void *p, size_type) {
    if (p) {
      cudaError_t err = cudaFree(p);
      if (cudaSuccess != err) {
        LOG_FATAL(cudaGetErrorString(err));
      }
    }
  }

  pointer address(reference x) const { return &x; }
  const_pointer address(const_reference x) const { return &x; }
  device_allocator<T> &operator=(const device_allocator &) { return *this; }
  void construct(pointer p, const T &val) { new ((T *)p) T(val); }
  void destroy(pointer p) { p->~T(); }

  size_type max_size() const { return size_t(-1); }

  template <class U> struct rebind { typedef device_allocator<U> other; };

  template <class U> device_allocator(const device_allocator<U> &) {}

  template <class U> device_allocator &operator=(const device_allocator<U> &) {
    return *this;
  }
};
