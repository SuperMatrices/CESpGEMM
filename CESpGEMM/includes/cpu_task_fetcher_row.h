#pragma once
#include"task_alloc.h"
#include<vector>
#include<functional>
#include"WritableStorage.h"
#include"Storage.h"
#include"computeMKL.h"
namespace CESpGEMM
{
template<typename Gs_t, bool SingleBlock>
struct CPU_TaskFetcher_by_row
{
  using EIdType = typename Gs_t::eidType;
  using ValType = typename Gs_t::valType;

  using fetch_t = TaskFetch<2, int>;
  using task_alloc_t = fetch_t::task_allocator_t;
  using wt_t = ComputeResultData<Gs_t, SingleBlock>;

  int m_rBid;
  int m_buffer;
  fetch_t pipeline_fetcher;
  Gs_t*gs;
  pthread_t m_thread;
  MKL_computer<Gs_t, SingleBlock> &mkl_computer;
  // CpuComupter<Gs_t> computer;

  CPU_TaskFetcher_by_row(task_alloc_t & ta, Gs_t * gs, MKL_computer<Gs_t, SingleBlock>&mklcomputer) : pipeline_fetcher(std::ref(ta), 0), m_rBid(0), m_buffer(0), mkl_computer(mklcomputer)
  {
    this->gs = gs;
    pthread_create(&m_thread, nullptr, computeRoute, this);
  }
  ~CPU_TaskFetcher_by_row(){
    pthread_join(m_thread, nullptr);
  }
  void change_row_and_buffer(int r,int buffer){
    m_rBid = r;
    m_buffer = buffer;
  }
  void route(){
    while(true){
      int cBid = pipeline_fetcher.fetch_work();
      // printf("CPU FETCHER FETCHED: %d !!\n", cBid);
      if(cBid==-1){
        break;
      }
      if(cBid < 0){
        int code = (-cBid)-10;
        int buffer = code&1, rowBid = code>>1;
        change_row_and_buffer(rowBid, buffer);
      }else{
        mkl_computer.compute(m_rBid, m_buffer);
      }
    }
  }
  static void* computeRoute(void*arg){
    CPU_TaskFetcher_by_row<Gs_t, SingleBlock>*_This=(CPU_TaskFetcher_by_row*)(arg);
    _This->route();
    return nullptr; 
  }
  

} ;
  
} // namespace CESpGEMM
