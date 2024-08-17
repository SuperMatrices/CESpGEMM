#pragma once
#include<iostream>
#include<vector>
#include<functional>
#include<semaphore.h>
#include<thread>
#include<chrono>
#include<pthread.h>
#include"profiler.h"
#include"helper.cuh"

namespace CESpGEMM{

struct emptyData{};

struct SymStage
{
  sem_t _avail;
  sem_t _empty;
  SymStage(){
    sem_init(&_avail, 0, 0);
    sem_init(&_empty, 0, 1);
  }
  void post(){
    SEM_POST(&_avail);
  }
  void require(){
    SEM_WAIT(&_avail);
  }
  void wait_empty(){
    SEM_WAIT(&_empty);
  }
  void post_empty(){
    SEM_POST(&_empty);
  }
} ;

struct NullStage{
  NullStage(){
  }
  void post(){
  }
  void require(){
  }
  void wait_empty(){
  }
  void post_empty(){
  }
} ;

template<typename TidType, int nBuffs>
struct StageTaskBuffer
{
  static constexpr int S = 4;
  using q_t = std::array<TidType, S>;
  
  q_t task_queue_;
  uint8_t start_, end_;
  sem_t nEmp_, nFil_;
  StageTaskBuffer(){
    static_assert(nBuffs<S);
    // static_assert(std::__or_v < std::is_same_v<Stage_t, SymStage> , std::is_same_v<Stage_t, NullStage> >);
    start_ = 0;
    end_ = 0;
    sem_init(&nEmp_, 0, nBuffs);
    sem_init(&nFil_, 0, 0);
  }
  void acquire(TidType *ret){
    SEM_WAIT(&nFil_);
    *ret = task_queue_[start_];
    start_ = (start_+1)&(S-1);
    SEM_POST(&nEmp_);
  }
  void post(const TidType &src){
    SEM_WAIT(&nEmp_);
    task_queue_[end_] = src;
    end_ = (end_ + 1)&(S-1);
    SEM_POST(&nFil_);
  }
} ;

template<typename TidType>
struct OneSideBuffer
{
  std::vector<TidType>tasks;
  //for input 
  OneSideBuffer(std::vector<TidType>&&v):tasks(std::move(v)){
    std::reverse(tasks.begin(),tasks.end());
  }

  //for output
  OneSideBuffer(int len){
    tasks.reserve(len);
  }
  //for output
  OneSideBuffer(){
    tasks.reserve(0);
  }

  void acquire(TidType*ret){
    *ret = tasks.back();
    tasks.pop_back();
  }
  void post(TidType ret){
    tasks.push_back(ret);
  }
  TidType* begin(){
    return &tasks[0];
  }
  TidType* end(){
    return &tasks.back() + 1;
  }

} ;


template<typename TidType>
struct EmptyTaskBuffer
{
  EmptyTaskBuffer(){
  }
  void acquire(TidType *ret){
  }
  void post(const TidType &src){
  }

} ;

//tasks are submited from the previous stage to the next stage 
template<typename PrevStage_t, typename NextStage_t, typename Recv_from_t, typename Send_to_t, typename InternalWorker>
struct PipeLineWorker
{
  static constexpr int nRep = 2;
  
  PrevStage_t *prev_stage[nRep];
  NextStage_t *next_stage[nRep];
  Recv_from_t *rBuffer;
  Send_to_t *sBuffer;

  InternalWorker worker;
  pthread_t m_pthread;

  static void* WorkProcess(void* arg){
    PipeLineWorker *pworker = reinterpret_cast<PipeLineWorker<PrevStage_t, NextStage_t, Recv_from_t, Send_to_t, InternalWorker>*>(arg);
    Map_Util &mp_inst = Map_Util::Instance();
    for(int buffer=0,task_id;;buffer^=1){
      // printf("%p,%s acquiring task\n", pworker, mp_inst.get(arg).c_str());
      pworker->rBuffer->acquire(&task_id);
      // printf("%p acquired %d\n", pworker, task_id);
      if(task_id == -1){
        printf("%s finish\n", mp_inst.get(arg).c_str());
        pworker->sBuffer->post(task_id);
        // is a member method "finish" needed?
        pworker->worker.finish();
        break;
      }

      // printf("%p,%s wait on prvStage[%d]=%p\n", pworker,mp_inst.get(arg).c_str(), buffer, pworker->prev_stage[buffer]);
      pworker->prev_stage[buffer] -> require();
      // printf("%p,%s wait_empty[%d] %p\n", pworker,mp_inst.get(arg).c_str(), buffer, &pworker->next_stage[buffer]);
      // printf("%p,%s wait_empty on nxtStage[%d]=%p\n", pworker,mp_inst.get(arg).c_str(), buffer, pworker->next_stage[buffer]);
      pworker->next_stage[buffer] -> wait_empty();
      
      // double t0 = profiler::Instance().get_tick_since_start();
      pworker->worker.work(task_id, buffer);
      // double t1 = profiler::Instance().get_tick_since_start();
      // profiler::Instance().worker_trace[(uintptr_t)pworker].push_back({t0,t1});

      // printf("%p,%s post on nxtStage[%d]=%p\n", pworker,mp_inst.get(arg).c_str(), buffer, pworker->next_stage[buffer]);
      pworker->next_stage[buffer] -> post();
      // printf("%p,%s post_empty[%d] %p\n", pworker,mp_inst.get(arg).c_str(), buffer, &pworker->prev_stage[buffer]);

      // printf("%p,%s post_empty on prvStage[%d]=%p\n", pworker,mp_inst.get(arg).c_str(), buffer, pworker->prev_stage[buffer]);
      pworker->prev_stage[buffer] -> post_empty();
      pworker->sBuffer->post(task_id);
    }
    return nullptr;
  }
  PipeLineWorker(InternalWorker &&wk, std::vector<PrevStage_t> &pv, std::vector<NextStage_t> &nt, Recv_from_t &rb, Send_to_t &sb): worker(std::move(wk)){
    CHK_ASSERT( pv.size() == nRep );
    CHK_ASSERT( nt.size() == nRep );
    for(int i=0;i<nRep;i++){
      prev_stage[i] = &pv[i];
    }
    for(int i=0;i<nRep;i++){
      next_stage[i] = &nt[i];
    }
    rBuffer = &rb;
    sBuffer = &sb;
    // printf("this %p\n", this);
    pthread_create(&m_pthread, nullptr, WorkProcess, this);
  }
  ~PipeLineWorker(){
    pthread_join(m_pthread, nullptr);
  }
} ;

// template<typename PrevData, typename NextData>
// struct TaskWorkerBindMem
// {
//   static constexpr int nRep = 2;
//   std::array< PrevData* , nRep > pData;
//   std::array< NextData* , nRep > nData;

//   TaskWorkerBindMem(std::vector<PrevData>&pd, std::vector<NextData>&nd){
//     for(int i=0;i<nRep;i++) pData[i] = &pd[i];
//     for(int i=0;i<nRep;i++) nData[i] = &nd[i];
//   }
// } ;

// template<typename NextData>
// struct TaskWorkerBindMem<emptyData,NextData>
// {
//   static constexpr int nRep = 2;
//   std::array< NextData* , nRep > nData;
//   TaskWorkerBindMem(std::vector<NextData>&nd){
//     for(int i=0;i<nRep;i++) nData[i] = &nd[i];
//   }
  
// } ;

// template<typename PrevData>
// struct TaskWorkerBindMem<PrevData,emptyData>
// {
//   static constexpr int nRep = 2;
//   std::array< PrevData* , nRep > pData;
//   TaskWorkerBindMem(std::vector<PrevData>&pd){
//     for(int i=0;i<nRep;i++) pData[i] = &pd[i];
//   }
// } ;

// template<>
// struct TaskWorkerBindMem<emptyData, emptyData>
// {
// } ;

}