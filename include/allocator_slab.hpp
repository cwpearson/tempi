#pragma once

#include <algorithm>
#include <cstdlib>
#include <vector>
#include <iostream>

#include <cuda_runtime.h>

#include "cuda_runtime.hpp"

template <typename T> class my_allocator {
public:
  typedef size_t size_type;
  typedef ptrdiff_t difference_type;
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef T &reference;
  typedef const T &const_reference;
  typedef T value_type;

private:
  struct Slab {
    std::vector<bool> avail;
    std::vector<void *> ptrs;

    /* all spot are taken
     */
    bool full() {
      if (avail.empty()) {
        return true;
      } else {
        return std::all_of(avail.begin(), avail.end(),
                           [](bool b) { return !b; });
      }
    }

    /* all spots are free
     */
    bool empty() {
      if (avail.empty()) {
        return false;
      } else {
        return std::all_of(avail.begin(), avail.end(),
                           [](bool b) { return b; });
      }
    }

    bool partial() {
      return std::any_of(avail.begin(), avail.end(), [](bool b) { return b; });
    }
  };

  std::vector<Slab> cache;

  /* retrieve an allocation for size i
   */
  void *get_slab(size_t i) {
    if (i >= cache.size()) {
      cache.resize(i + 1);
    }

    Slab &slab = cache[i];

    // if there is a pre-allocated available spot, mark it as full and return it
    if (slab.partial()) {
      // std::cerr << "preallocated\n";
      for (size_t j = 0; j < slab.avail.size(); ++j) {
        if (slab.avail[j]) {
          slab.avail[j] = false;
          return slab.ptrs[j];
        }
      }
      std::cerr << "UNREACHABLE\n";
      exit(1);
    } else { // otherwise, make a few one and mark it as full and reuturn
      void *newPtr;
      CUDA_RUNTIME(cudaMalloc(&newPtr, 1 << i));
      slab.avail.push_back(false);
      slab.ptrs.push_back(newPtr);
      return newPtr;
    }
  }

public:
  my_allocator() {}
  my_allocator(const my_allocator &) {}

  pointer allocate(size_type n, const void * = 0) {

    // find the slabs for smallest 2^x >= n
    int log2 = 64 - __builtin_clzll(n);
    // if input is not a power of two, this will be too small
    if (__builtin_popcount(n) != 1 && __builtin_popcount(n) != 0) {
      ++log2;
    }

    return (pointer)get_slab(log2);
  }

  void deallocate(void *p, size_type) {
    // search through all slabs for a matching pointer and mark it as available
    for (Slab &slab : cache) {
      if (!slab.empty()) {
        for (size_t i = 0; i < slab.ptrs.size(); ++i) {
          if (slab.ptrs[i] == p) {
            slab.avail[i] = true;
            return;
          }
        }
      }
    }
    std::cerr << "Tried to free memory not from this allocator\n";
    exit(1);
  }

  pointer address(reference x) const { return &x; }
  const_pointer address(const_reference x) const { return &x; }
  my_allocator<T> &operator=(const my_allocator &) { return *this; }
  void construct(pointer p, const T &val) { new ((T *)p) T(val); }
  void destroy(pointer p) { p->~T(); }

  size_type max_size() const { return size_t(-1); }

  template <class U> struct rebind { typedef my_allocator<U> other; };

  template <class U> my_allocator(const my_allocator<U> &) {}

  template <class U> my_allocator &operator=(const my_allocator<U> &) {
    return *this;
  }
};

extern my_allocator<char> testAllocator;