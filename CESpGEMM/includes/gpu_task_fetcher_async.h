#pragma once
#include"task_alloc.h"
#include<functional>
#include"computeGPUAsync.h"
#include"profiler.h"
#include<vector>

namespace CESpGEMM
{
#ifndef NUM_DEVICES
#define NUM_DEVICES 1
#endif


template<class Gs_t, bool SingleBlock>
struct GPUTaskFetcherAsync
{
  using EIdType = typename Gs_t::eidType;
  using ValType = typename Gs_t::valType;
  using fetch_t = TaskFetch<2,int>;
  using task_alloc_t = TaskAlloc<2, int>;
  using rd_t = FlopData<EIdType>;
  using wt_t = ComputeResultData<Gs_t, SingleBlock>;
  using gpu_computer = GpuComputerAsync<Gs_t, SingleBlock>;

  fetch_t pipeline_fetcher;
  int m_rBid;
  int m_buffer;
  rd_t * rd[2];
  gpu_computer *gpu_worker;
  pthread_t worker_thread;
  GPUTaskFetcherAsync(task_alloc_t & pipeline_allocator, gpu_computer & gpu_worker, std::vector<rd_t> &read_data):
    pipeline_fetcher(std::ref(pipeline_allocator), 1),
    m_rBid(0), m_buffer(0)
  {
    CHK_ASSERT(read_data.size()==2);
    this->gpu_worker = &gpu_worker;
    for(int i=0;i<2;i++){
      rd[i] = &read_data[i];
    }
    pthread_create(&worker_thread, nullptr, compute_route, this);
  }
  ~GPUTaskFetcherAsync(){
    pthread_join(worker_thread, nullptr);
  }
  void change_row_and_buffer(int rBid, int gBuffer){
    m_rBid = rBid;
    m_buffer = gBuffer;
    // FlopData<EIdType> *rd_data = rd[gBuffer];
    // IdxType cid = rd_data->cBlockId.front();
    // EIdType max_flop = rd_data->flops.at(cid);
    // gpu_worker->realloc_size(max_flop);
    // printf("rowid: %d, maxflop=%lld\n", rBid, max_flop);
  }
  uint8_t subthread_handle_task_id(int task_id){
    if(task_id < 0){
      int code = (-task_id) - 10;
      int buffer = code & 1, rowBid = code>>1;
      // printf("computeGPUAsync: change Row: %d,%d\n", rowBid, buffer);
      change_row_and_buffer(rowBid, buffer);
      return 0;
    }
    else{
      profiler::Instance().h2d_time += profiler::Instance().timer_h2d.consume();
      gpu_worker->addGpuTask(m_rBid, task_id, m_buffer);
      return TAG_H2D;
    }
  }
  static void*compute_route(void*arg){
    using namespace std::chrono_literals;
    GPUTaskFetcherAsync *This = static_cast<GPUTaskFetcherAsync<Gs_t, SingleBlock>*>(arg);
    gpu_computer &gworker = *This->gpu_worker;
    GPUControllingBlock<EIdType> &gcb = gworker.gcb;
    int task_id;
    bool all_received=0;
    int rr=0;
    profiler &prf = profiler::Instance();
    auto t_start = std::chrono::steady_clock::now();
    while(true){
      uint8_t tag = 1<<rr;
      (rr += 1) &= 3;
      // std::this_thread::sleep_for(500ms);
      // printf(">>>free_tags = %d, gcb=%p\n", free_tags, &gcb);
      if(gcb.free_state & tag) {
        if(tag == TAG_H2D && !all_received){
          bool local_buffer_id = gworker.counter & 1;
          if(gcb.buffer_status[local_buffer_id] & TAG_KNL) continue;
          if(gcb.knl_info.pending) continue; //CHECK, but after checking, the host function may set knl_info.pending true
          bool is_empty = This->pipeline_fetcher.empty_or_fetch(task_id);
          // printf("is_empty=%d, task_id=%d\n", is_empty, task_id);
          if(is_empty) continue;
          if(task_id == -1){
            all_received = true;
            continue;
          }
          This->subthread_handle_task_id(task_id);
        }
        else if(tag == TAG_KNL){
          // printf("CHK KNL LAUNCH: knlpending: %d\n", gcb.knl_info.pending);
          if(!gcb.knl_info.pending) continue;
          int local_buffer_id = gcb.knl_info.local_buffer_id;
          // printf("CHK KNL LAUNCH: bufferstat[%d]=%d\n", local_buffer_id, gcb.buffer_status[local_buffer_id]);
          if(gcb.buffer_status[local_buffer_id] & (TAG_H2D|TAG_DAT|TAG_PTR) ) continue;
          if(gcb.ptr_info.pending) continue;
          prf.kernel_time += prf.timer_knl.consume();
          gworker.doKernel();
        }
        else if(tag == TAG_PTR){
          if(!gcb.ptr_info.pending) continue;
          int local_buffer_id = gcb.ptr_info.local_buffer_id;
          if(gcb.buffer_status[local_buffer_id] & TAG_KNL) continue;
          if(gcb.data_info.pending) continue;
          prf.d2h_time += prf.timer_d2h.consume();
          gworker.doPtrD2H();
        }
        else if(tag == TAG_DAT){
          if(!gcb.data_info.pending) continue;
          int local_buffer_id = gcb.data_info.local_buffer_id;
          if(gcb.buffer_status[local_buffer_id] & TAG_KNL) continue;
          prf.d2h_time += prf.timer_d2h.consume();
          gworker.doDataD2H();
        }
      }
      if(all_received && gcb.free_state == TAG_ALL && !gcb.knl_info.pending && !gcb.ptr_info.pending && !gcb.data_info.pending) {
        printf("GPUFETCHER QUIT!\n");
        // std::this_thread::sleep_for(500ms);
        break;
      }
    }
    prf.h2d_time += prf.timer_h2d.consume();
    prf.kernel_time += prf.timer_knl.consume();
    prf.d2h_time += prf.timer_d2h.consume();
    auto t_end = std::chrono::steady_clock::now();
    prf.gpu_computer_time += get_chrono_ms(t_start, t_end);
    return nullptr;
  }
} ;

template
struct GPUTaskFetcherAsync<default_gs_t, false>;
template
struct GPUTaskFetcherAsync<default_gs_t, true>;
template
struct GPUTaskFetcherAsync<large_gs_t, false>;
template
struct GPUTaskFetcherAsync<large_gs_t, true>;
  
} // namespace CESpGEMM
