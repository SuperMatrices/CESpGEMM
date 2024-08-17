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
template<typename Gs_t, bool Singleblock, size_t Alpha, int Ratio_permil>
struct ComputeDispatcher
{
  static constexpr size_t cpu_threshold = Alpha * 32;
  static constexpr int gpu_ratio = Ratio_permil;

  using EIdType = typename Gs_t::eidType;
  using ValType = typename Gs_t::valType;



  using Rd_t = FlopData<EIdType>;
  using Wt_t = ComputeResultData<Gs_t, Singleblock>;
  using Tsk_alloc_t = TaskAlloc<2, int>; //-1

  Gs_t *gs;
  std::vector<Rd_t *> rd;
  std::vector<Wt_t *> wt;
  Tsk_alloc_t *ta;

  ComputeDispatcher(Gs_t *gs, std::vector<Rd_t>&from_data, std::vector<std::unique_ptr<Wt_t>>&to_data, Tsk_alloc_t &task_allocator) : gs(gs), rd(from_data.size()), wt(to_data.size()){
    for(int i=0;i<(int)from_data.size();i++){
      rd[i]=&from_data[i];
    }
    for(int i=0;i<(int)to_data.size();i++){
      wt[i]=to_data[i].get();
    }
    ta = &task_allocator;
  }
  ComputeDispatcher(ComputeDispatcher && rhs){
    gs = rhs.gs;  rhs.gs = nullptr;
    ta = rhs.ta;  rhs.ta = nullptr;
    rd = std::move(rhs.rd);
    wt = std::move(rhs.wt);
  }

  enum BlockType{
    kCPU, kGPU
  } ;

  BlockType getType(EIdType flops){
    if(flops <= cpu_threshold) return kCPU;
    else if(flops > gs->gpu_flop_thresh) return kGPU;
    else{
      int slaveId = ta->wait_either_deployed();
      return slaveId == 0 ? kCPU: kGPU;
    }
    // return kCPU;
    // if(flops < 1000)
  }


  // void work_by_row_col(int tid, int buffer){
  //   printf("compute work at (%d,%d)\n", tid, buffer);
  //   int NBB = gs->numBlocksB;
  //   Rd_t *read_data = rd[buffer];
  //   for(int i=0;i<2;i++){
  //     ta->push_at(i, -10-(tid << 1 | buffer)); //change row
  //   }
  //   for(int i=0;i<NBB;i++){
  //     wt[buffer]->block_status[i] = 0;
  //   }
    
  //   for(int idx: read_data->cBlockId){
  //     EIdType flops = read_data->flops[idx];
  //     if(flops == 0){
  //       wt[buffer]->block_status[idx] = 255;
  //       profiler::Instance().skipped_col_block += 1;
  //       continue;
  //     }
  //     int selected_device = getType(flops) == kGPU;
  //     if(selected_device) profiler::Instance().blocks_gpu++;
  //     else profiler::Instance().blocks_cpu++; 
  //     // printf("push work to device %d: rid=%d, cid=%d, flops=%d\n", selected_device, tid, idx, flops);
  //     ta->push_at(selected_device, idx);
  //   }
  //   printf("compute worker: wait all deployed\n");
	// 	ta->wait_all_deployed();
  //   printf("compute worker: all deployed, ok\n");
  // }

  BlockType get_rb_type(unsigned long long flops){
    // return kGPU;
    // printf("at get_rb_type: get gpuflop_thresh=%lld, this_flop=%lld\n", gs->gpu_flop_thresh, flops);
    if(flops > gs->gpu_flop_thresh){
      return kGPU;
    }
    if(flops < Alpha * 32){
      return kCPU;
    }
    int slaveId = ta->wait_either_deployed();
    return slaveId == 0 ? kCPU:kGPU;
  }

  void work_by_row(int tid, int buffer){
    ull flop = gs->block_flops[tid];
    ComputeResultData<Gs_t, Singleblock> &res = *wt[buffer];
    res.setBlockStatusAll(0);
    for(int i=0;i<2;i++){
      ta->push_at(i, -10-(tid << 1 | buffer)); //change row
    }
    BlockType btype = get_rb_type(flop);
    // printf("BLOCKTYPE=%s\n", btype==kCPU?"C":"G");
    if(btype == kCPU){
      ta->push_at(0/*workerid*/, 0/*cbid*/);
      profiler::Instance().blocks_cpu ++;
    }
    else{
      Rd_t &read_data = *rd.at(buffer);
      for(int idx: read_data.cBlockId){
        EIdType cb_nnz = read_data.cbNNZ[idx];
        if(cb_nnz == 0){
          res.setBlockStatus(idx,255);
          profiler::Instance().skipped_col_block += 1;
          continue;
        }
        ta->push_at(1, idx);
        // printf("rbid=%d, push idx%d to GPU:\n", tid, idx);
      }
      profiler::Instance().blocks_gpu++;
    }
  }
  void work(int tid, int buffer){
    work_by_row(tid, buffer);
    ta->wait_all_deployed();
    printf("dispatcher %d,%d all deployed\n", tid, buffer);
  }

  void finish(){
    printf("compute worker: finish all\n");
    // ta->finish_all();
    int nworkers = Tsk_alloc_t::nworkers;
    for(int i=0;i<nworkers;i++){
      ta->push_at(i, -1);
    }
  }
} ;
}//namespace CESpGEMM