//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include "counters.hpp"
#include "logging.hpp"
#include "numeric.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <vector>

template <typename T, typename Allocator> class SlabAllocator {
public:
  typedef typename Allocator::size_type size_type;
  typedef ptrdiff_t difference_type;
  typedef T *pointer;
  typedef const T *const_pointer;
  typedef T &reference;
  typedef const T &const_reference;
  typedef T value_type;

private:
  counters::Allocator &counters_;

  /*! \brief a pool of allocations of a particular size
   */
  struct Pool {
    std::vector<bool> avail;
    std::vector<void *> ptrs;

    /* all requests in the pool are full
     */
    bool full() const noexcept {
      if (avail.empty()) {
        return true;
      } else {
        return std::all_of(avail.begin(), avail.end(),
                           [](bool b) { return !b; });
      }
    }

    /* all requests in the pool are empty
     */
    bool empty() const noexcept {
      if (avail.empty()) {
        return false;
      } else {
        return std::all_of(avail.begin(), avail.end(),
                           [](bool b) { return b; });
      }
    }

    /* at least one spot in the pool is available
     */
    bool not_full() const noexcept {
      return std::any_of(avail.begin(), avail.end(), [](bool b) { return b; });
    }

    size_t size() const noexcept {
      assert(avail.size() == ptrs.size());
      return avail.size();
    }

    void clear() {
      avail.clear();
      ptrs.clear();
    }
  };

  // pools for allocations of different sizes
  std::vector<Pool> pools_;

  // retrieve the pool that serves allocations of size `n`
  Pool &get_pool_for_size(size_t n) {
    // find the slabs for smallest 2^x >= n
    const unsigned log2 = log2_ceil(n);

    if (log2 >= pools_.size()) {
      pools_.resize(log2 + 1);
    }

    return pools_[log2];
  }

  /* return the size of the allocation backing a request for `n` bytes
   */
  size_t alloc_size_for(size_t n) const noexcept {
    return 1ull << log2_ceil(n);
  }

  /* serve an allocation of size n
   */
  void *alloc_request(size_t n) {

    if (0 == n) {
      return nullptr;
    }

    Pool &pool = get_pool_for_size(n);

    // if there is a pre-allocated available spot, mark it as full and return it
    for (size_t j = 0; j < pool.size(); ++j) {
      if (pool.avail[j]) {
        pool.avail[j] = false;
        return pool.ptrs[j];
      }
    }
    { // otherwise, make a few one and mark it as full and reuturn
      const size_t allocSize = alloc_size_for(n);
      LOG_SPEW("calling allocator for " << allocSize << " to serve request of " << n);
      void *newPtr = allocator.allocate(allocSize);
#ifdef TEMPI_ENABLE_COUNTERS
      counters_.CURRENT_USAGE += allocSize;
      counters_.MAX_USAGE =
          std::max(counters_.MAX_USAGE, counters_.CURRENT_USAGE);
#endif

      pool.avail.push_back(false);
      pool.ptrs.push_back(newPtr);
      return newPtr;
    }
  }

public:
  SlabAllocator(counters::Allocator &counters) : counters_(counters) {}
  SlabAllocator(const SlabAllocator &) = default;

  /* free all memory in the pools
   */
  void release_all() {
    for (Pool &pool : pools_) {
      for (size_t i = 0; i < pool.size(); ++i) {
        const size_t allocSize = size_t(1) << i;
        allocator.deallocate(pool.ptrs[i], allocSize);
      }
      pool.clear();
    }
  }

  ~SlabAllocator() { release_all(); }

  pointer allocate(size_type n, const void * = 0) {
#ifdef TEMPI_ENABLE_COUNTERS
    counters_.NUM_REQUESTS++;
#endif
    return (pointer)alloc_request(n);
  }

  void deallocate(void *p, size_type n) {
#ifdef TEMPI_ENABLE_COUNTERS
    counters_.NUM_RELEASES++;
#endif
    if (0 == n) {
      return;
    }
    // retrieve the pool this allocation must have come from
    Pool &pool = get_pool_for_size(n);

    // find the pointer and mark it as available
    for (size_t i = 0; i < pool.size(); ++i) {
      if (pool.ptrs[i] == p) {
        pool.avail[i] = true;
        return;
      }
    }
    LOG_FATAL("Tried to free memory not from this allocator. Please report this.");
  }

  pointer address(reference x) const { return &x; }
  const_pointer address(const_reference x) const { return &x; }
  SlabAllocator<T, Allocator> &operator=(const SlabAllocator &) {
    return *this;
  }
  void construct(pointer p, const T &val) { new ((T *)p) T(val); }
  void destroy(pointer p) { p->~T(); }

  size_type max_size() const { return size_t(-1); }

  template <typename U, typename A> struct rebind {
    typedef SlabAllocator<U, A> other;
  };

  template <typename U, typename A>
  SlabAllocator(const SlabAllocator<U, A> &) {}

  template <typename U, typename A>
  SlabAllocator &operator=(const SlabAllocator<U, A> &) {
    return *this;
  }

private:
  Allocator allocator;
};
