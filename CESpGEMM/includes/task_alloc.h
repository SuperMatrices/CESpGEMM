#pragma once
#include<iostream>
#include<queue>
#include<array>
#include<atomic>
#include"ThreadSafeQueue.h"
#include<thread>

namespace CESpGEMM{

template<int nSlaves, typename TidType>
struct TaskAlloc
{
  static constexpr int nworkers = nSlaves;
  std::array<ThreadSafeQueue<TidType>, nSlaves> q ;
  std::array<std::atomic_bool,  nSlaves> running ;
  void push_at(int worker_id, TidType task_id){
    q[worker_id].enque(task_id);
  }
  TidType pop_at(int worker_id){
    return q[worker_id].pop_trigger_variable(running[worker_id]);
  }
  bool empty_or_pop_at(int worker_id, TidType&res){
    return q[worker_id].empty_or_pop(res);
  }

  int wait_either_deployed(){
    while(true){
      for(int i=0;i<nSlaves;i++){
        if(q[i].emptied()) return i;
        // printf("no %d~\n",i);
      }
      // std::this_thread::yield();
      // printf("??\n");
    }
  }

	void wait_all_deployed(){
    using namespace std::chrono_literals;
		for(int i=0;i<nSlaves;i++){
			while(!q[i].emptied()){
        std::this_thread::yield();
      }
		}
	}

  TaskAlloc(TaskAlloc&&t) noexcept{
    for(int i=0;i<nSlaves;i++){
      q[i] = std::move( t.q[i] );
      running[i] = t.running[i];
    }
  }
  TaskAlloc(){    
  }
} ;


//StageTaskBuffer is used to submit tasks to the pipeline workers
// template<int nSlaves, typename TidType, TidType endVal, typename StageTaskBuffer>
// struct PipeTaskFetch
// {
//   using task_allocator_t = TaskAlloc<nSlaves, TidType, endVal>;
//   using queue_t = ThreadSafeQueue<TidType>;
//   using interface_task_buffer_t = StageTaskBuffer<TidType, 2> ;
//   task_allocator_t * _ta;
//   interface_task_buffer_t * sBuffer;
//   int _wid;

//   PipeTaskFetch(task_allocator_t &ta, int worker_id, interface_task_buffer_t &task_buffer):_ta(&ta), _wid(worker_id), sBuffer(&task_buffer){
    
//   }
  
//   void do_work(){
//     while(true){
//       TidType task_id = ta.pop_at(_wid);
//       if(task_id == endVal) {
//         // do some work before exit.
//         sBuffer->post(endVal);
//         ta->running[_wid] = false;
//         break;
//       }
//       sBuffer->post(task_id);
//     }
//   }
// } ;


template<int nSlaves, typename TidType>
struct TaskFetch
{
  using task_allocator_t = TaskAlloc<nSlaves, TidType>;
  using queue_t = ThreadSafeQueue<TidType>;
  task_allocator_t * _ta;
  int _wid;

  TaskFetch(task_allocator_t &ta, int worker_id): _ta(&ta), _wid(worker_id){
    
  }
  bool empty_or_fetch(TidType&t){
    return _ta->empty_or_pop_at(_wid, t);
  }
  TidType fetch_work(){
    TidType task_id = _ta->pop_at(_wid);
    return task_id;
    // while(true){
    //   TidType task_id = ta.pop_at(_wid);
    //   if(task_id == endVal) {
    //     // do some work before exit.
    //     sBuffer->post(endVal);
    //     ta->running[_wid] = false;
    //     break;
    //   }
    //   sBuffer->post(task_id);
    // }
  }
} ;

}