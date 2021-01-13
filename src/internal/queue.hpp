//          Copyright Carl Pearson 2020 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE or copy at
//    https://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <queue>
#include <mutex>
#include <atomic>
#include <cassert>

/* https://codetrips.com/2020/07/26/modern-c-writing-a-thread-safe-queue/
 */
template <typename T> class Queue {
  std::queue<T> queue_;
  mutable std::mutex mutex_;

public:
  Queue() = default;
  Queue(const Queue<T> &) = delete;
  Queue &operator=(const Queue<T> &) = delete;

  Queue(Queue<T> &&other) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_ = std::move(other.queue_);
  }

  virtual ~Queue() {}

  unsigned long size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

  T pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    assert(!queue_.empty());
    T tmp = queue_.front();
    queue_.pop();
    return tmp;
  }

  void push(const T &item) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(item);
  }
};