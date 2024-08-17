#pragma once
#include<queue>
#include<mutex>
#include<condition_variable>

namespace CESpGEMM
{
template<typename T>
struct ThreadSafeQueue
{
  std::queue<T> _q;
  mutable std::mutex _mut;
  std::condition_variable _cv;
  ThreadSafeQueue()=default;

  void enque(T x){
    std::lock_guard<std::mutex> lk(_mut);
    _q.push(x);
    _cv.notify_one();
  }

  T pop(){
    std::unique_lock<std::mutex> lk(_mut);
    while(_q.empty()) _cv.wait(lk);
    auto ret = std::move(_q.front());
    _q.pop();
    return ret;
  }

  bool empty_or_pop(T&res){
    std::lock_guard<std::mutex>lk(_mut);
    if(_q.empty()){
      return true;
    }
    else{
      res = std::move(_q.front());
      _q.pop();
      return false;
    }
  }

  T pop_trigger_variable(std::atomic_bool & b){
    std::unique_lock<std::mutex> lk(_mut);
    while(_q.empty()) _cv.wait(lk);
    auto ret = std::move(_q.front());
    b=true;
    _q.pop();
    return ret;
  }
  
  size_t size(){return _q.size();}
  bool emptied(){
    std::lock_guard<std::mutex> lk(_mut);
    bool res = _q.empty();
    return res;
  }

} ;

} // namespace CESpGEMM
