#pragma once
#include"task_alloc.h"
#include<functional>
#include"computeGPUAsync.h"
#include"profiler.h"
#include<vector>
#include<nvml.h>
namespace CESpGEMM
{

template<class Gs_t, bool SingleBlock>
struct GPUTaskFetcherAsyncMGPU
{
    using EIdType = typename Gs_t::eidType;
    using ValType = typename Gs_t::valType;
    static constexpr int num_gpus = NUM_DEVICES;
    static constexpr int num_compute_workers = 1 + num_gpus;
    using fetch_t = TaskFetch<num_compute_workers, int>;
    using task_alloc_t = TaskAlloc<num_compute_workers, int>;
    using rd_t = FlopData<EIdType>;
    using wt_t = ComputeResultData<Gs_t, SingleBlock>;
    using gpu_computer_t = GpuComputerAsync<Gs_t, SingleBlock>;
    
    std::vector<fetch_t> pipeline_fetcher;
    rd_t * rd[2];
    std::vector<gpu_computer_t*> gpu_workers;
    std::vector<std::thread> worker_threads;
    
    GPUTaskFetcherAsyncMGPU(task_alloc_t & pipeline_allocator, std::vector<std::unique_ptr<gpu_computer_t>> & gpu_computer, std::vector<rd_t>&read_data)
    {
        CHK_ASSERT(read_data.size() == 2);
        CHK_ASSERT(gpu_computer.size() == num_gpus);
        pipeline_fetcher.reserve(num_gpus);
        for(int i=0;i<num_gpus;i++){
            pipeline_fetcher.emplace_back(std::ref(pipeline_allocator), i + 1);
        }
        for(int i=0;i<num_gpus;i++){
            gpu_workers.push_back(gpu_computer[i].get());
        }
        for(int i=0;i<2;i++) rd[i] = &read_data[i];
        worker_threads.reserve(num_gpus);
        for(int i=0;i<num_gpus;i++){
            worker_threads.emplace_back(compute_route, this, i);
        }
    }
    ~GPUTaskFetcherAsyncMGPU(){
        for(int i=0;i<num_gpus;i++){
            worker_threads[i].join();
        }
    }
    uint8_t subthread_handle_task_id(int task_id, int device_id, int rBid, int gBuffer){
        if(task_id < 0){
            throw std::runtime_error("subthread_handle_task_id");
            return 0;
        } else{
            if(this->gpu_workers[device_id]->addGpuTask(rBid, task_id, gBuffer)){
                pipeline_fetcher[device_id]._ta->notifier.notify();
            }
            return TAG_H2D;
        }
    }
    static void compute_route(GPUTaskFetcherAsyncMGPU *This, const int device_id){
        printf("GPU Compute Route (%lx, LWP %d)\n", pthread_self(), gettid());
        using namespace std::chrono_literals;
        int current_rBid = 0;
        int current_gBuffer = 0;
        gpu_computer_t &gworker = *This->gpu_workers[device_id];
        gworker.set_call_back_finish_counter(This->pipeline_fetcher.at(device_id).finish_counter);
        GPUControllingBlock<EIdType> &gcb = gworker.gcb;
        gpu_profile_t *gpu_prof = gworker.prof;
        fetch_t &fetcher = This->pipeline_fetcher[device_id];
        int task_id;
        bool all_received = 0;
        int rr = 0;
        auto t_start = std::chrono::steady_clock::now();
        try{
            while(true){
                uint8_t tag = 1<<rr;
                (rr+=1) &= 3;
                if( ! (gcb.free_state & tag) ) continue;
                if(tag == TAG_H2D && !all_received){
                    bool local_buffer_id = gworker.counter & 1;
                    if(gcb.buffer_status[local_buffer_id] & TAG_KNL) continue;
                    if(gcb.knl_info.pending) continue;
                    bool is_empty = fetcher.empty_or_fetch(task_id);
                    if(is_empty) continue;
                    if(task_id == -1){
                        all_received = true;
                        continue;
                    }
                    if(task_id < 0){
                        auto [rowBid, buffer] = decode_row_buffer(task_id);
                        current_rBid = rowBid;
                        current_gBuffer = buffer;
                        continue;
                    }
                    gpu_prof->h2d_time += gpu_prof->timer_h2d.consume();
                    This->subthread_handle_task_id(task_id, device_id, current_rBid, current_gBuffer);
                }
                else if(tag == TAG_KNL){
                    if(!gcb.knl_info.pending) continue;
                    int local_buffer_id = gcb.knl_info.local_buffer_id;
                    if(gcb.buffer_status[local_buffer_id] & (TAG_H2D | TAG_DAT | TAG_PTR)) continue;
                    if(gcb.ptr_info.pending) continue;
                    gpu_prof->kernel_time += gpu_prof->timer_knl.consume();
                    if(gworker.doKernel()) fetcher._ta->notifier.notify();
                }
                else if(tag == TAG_PTR){
                    if(!gcb.ptr_info.pending) continue;
                    int local_buffer_id = gcb.ptr_info.local_buffer_id;
                    if(gcb.buffer_status[local_buffer_id] & TAG_KNL) continue;
                    if(gcb.data_info.pending) continue;
                    gpu_prof->d2h_time += gpu_prof->timer_d2h.consume();
                    if(gworker.doPtrD2H()) fetcher._ta->notifier.notify();
                }
                else if(tag == TAG_DAT){
                    if(!gcb.data_info.pending) continue;
                    int local_buffer_id = gcb.data_info.local_buffer_id;
                    if(gcb.buffer_status[local_buffer_id] & TAG_KNL) continue;
                    gpu_prof->d2h_time += gpu_prof->timer_d2h.consume();
                    if(gworker.doDataD2H()) fetcher._ta->notifier.notify();
                }
                if(all_received && gcb.free_state == TAG_ALL && !gcb.knl_info.pending && !gcb.ptr_info.pending && !gcb.data_info.pending)
                {
                    break;
                }
            }
        }
        catch(const std::exception &e){
            printf("in %s:%d, caught exception in device %d!: %s\n", __FILE__, __LINE__, device_id, e.what());
        }
        gpu_prof->h2d_time += gpu_prof->timer_h2d.consume();
        gpu_prof->kernel_time += gpu_prof->timer_knl.consume();
        gpu_prof->d2h_time += gpu_prof->timer_d2h.consume();
        auto t_end = std::chrono::steady_clock::now();
        gpu_prof->gpu_computer_time += get_chrono_ms(t_start, t_end);
    }
} ;

} //namespace CESpGEMM