#pragma once
#include<iostream>
#include<queue>
#include<array>
#include<atomic>
#include"ThreadSafeQueue.h"
#include"ErrorNotifier.h"
#include<thread>
#include<tuple>


namespace CESpGEMM{

inline int encode_row_buffer(int rowBid, int bufferId){
  return -10 - (rowBid<<1) - bufferId;
}
inline std::tuple<int,int> decode_row_buffer(int task_id){
  int code = - task_id - 10;
  return {code >> 1, code & 1};
}


template<int nSlaves, typename TidType>
struct TaskAlloc
{
  static constexpr int nworkers = nSlaves;
  std::array<ThreadSafeQueue<TidType>, nSlaves> q ;
  std::array<std::atomic_llong, nworkers> num_deployed, num_finished;
  std::array<std::atomic_bool,  nSlaves> running ;
  ErrorNotifier notifier;

  void push_at(int worker_id, TidType task_id, bool is_task = true){
    q[worker_id].enque(task_id);
    if(is_task) num_deployed[worker_id].fetch_add(1, std::memory_order::memory_order_relaxed);
  }
  bool is_finished(int worker_id){
    long long dep = num_deployed[worker_id].load(std::memory_order::memory_order_relaxed);
    long long fin = num_finished[worker_id].load(std::memory_order::memory_order_relaxed);
    return dep == fin;
  }

  TidType pop_at(int worker_id){
    return q[worker_id].pop_trigger_variable(running[worker_id]);
  }
  bool empty_or_pop_at(int worker_id, TidType&res){
    return q[worker_id].empty_or_pop(res);
  }

	int wait_all_deployed(){
    using namespace std::chrono_literals;
  for(int i=0;i<nSlaves;i++){
			while(!q[i].emptied()){
        std::this_thread::yield();
        if(notifier.has_error()) return 1;
      }
		}
    return 0;
	}

  TaskAlloc(TaskAlloc&&)=delete;

  TaskAlloc(){
    for(int i=0;i<nworkers;i++){
      num_deployed[i].store(0, std::memory_order::memory_order_seq_cst);
      num_finished[i].store(0, std::memory_order::memory_order_seq_cst);
    }
  }
} ;

template<int nSlaves, typename TidType>
struct TaskFetch
{
  using task_allocator_t = TaskAlloc<nSlaves, TidType>;
  using queue_t = ThreadSafeQueue<TidType>;
  task_allocator_t * _ta;
  std::atomic_llong * finish_counter;
  int _wid;


  TaskFetch(task_allocator_t &ta, int worker_id): _ta(&ta), _wid(worker_id){
    finish_counter = &ta.num_finished[worker_id];
  }
  bool empty_or_fetch(TidType&t){
    return _ta->empty_or_pop_at(_wid, t);
  }
  void finish_task(){
    // _ta->num_finished[_wid].fetch_add(1, std::memory_order::memory_order_relaxed);
    finish_counter->fetch_add(1, std::memory_order::memory_order_relaxed);
  }

  TidType fetch_work(){
    TidType task_id = _ta->pop_at(_wid);
    return task_id;
  }
} ;

}