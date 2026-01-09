#pragma once
#include<mutex>
#include<atomic>


namespace CESpGEMM{
class ErrorNotifier{
public: 
  void notify(){
    std::lock_guard<std::mutex> lk(m_); 
    if(!flag_){
      flag_ = true;
    }
  }
  bool has_error() const {
    return flag_;
  }
private:
  mutable std::mutex m_;
  std::atomic_bool flag_{false};
} ;
}