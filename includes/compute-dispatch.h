#pragma once
#include<iostream>
#include<vector>
#include"task_alloc.h"
#include"Storage.h"
#include"WritableStorage.h"
#include"profiler.h"
#include"mklManage.h"

namespace CESpGEMM
{
template<typename Gs_t, bool Singleblock, size_t Alpha>
struct ComputeDispatcher
{
  static constexpr size_t cpu_threshold = Alpha * 32;
  static constexpr int num_gpus = NUM_DEVICES;
  static constexpr int num_workers = 1 + num_gpus;

  using EIdType = typename Gs_t::eidType;
  using ValType = typename Gs_t::valType;

  using Rd_t = FlopData<EIdType>;
  using Wt_t = ComputeResultData<Gs_t, Singleblock>;
  using Tsk_alloc_t = TaskAlloc<num_workers, int>; //-1

  const int group_size;
  Gs_t *gs;
  std::vector<Rd_t *> rd;
  std::vector<Wt_t *> wt;
  Tsk_alloc_t *ta;
  int current_gpu_id;


  ComputeDispatcher(Gs_t *gs, std::vector<Rd_t>&from_data, std::vector<std::unique_ptr<Wt_t>>&to_data, Tsk_alloc_t &task_allocator, int group_size);
  ComputeDispatcher(ComputeDispatcher && rhs);


  int wait_any_deployed(int begin_id, int end_id);
  int wait_any_finished(int begin_id, int end_id);
  int get_rb_type(unsigned long long flops);
  int get_rb_type_try_gpu(ull flops);
  int work(int tid, int buffer);
  void finish(){
    // printf("compute worker: finish all\n");
    // ta->finish_all();
    int nworkers = Tsk_alloc_t::nworkers;
    for(int i=0;i<nworkers;i++){
      ta->push_at(i, -1);
    }
  }
} ;
}//namespace CESpGEMM