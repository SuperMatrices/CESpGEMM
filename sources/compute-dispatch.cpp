#include"compute-dispatch.h"
#include<vector>
#include"task_alloc.h"
#include"Storage.h"
#include"WritableStorage.h"
#include"profiler.h"
#include"mklManage.h"
#include"logger.h"


namespace CESpGEMM
{

// Constructor: Initialize dispatcher with global storage, read/write buffers, and task allocator
template<typename Gs_t, bool Singleblock, size_t Alpha>
ComputeDispatcher<Gs_t, Singleblock, Alpha>::ComputeDispatcher(Gs_t *gs, std::vector<Rd_t>&from_data, std::vector<std::unique_ptr<Wt_t>>&to_data, Tsk_alloc_t &task_allocator, int group_size) :
    gs(gs), rd(from_data.size()), wt(to_data.size()), group_size(group_size)
{
    for(int i=0;i<(int)from_data.size();i++){
        rd[i]=&from_data[i];
    }
    for(int i=0;i<(int)to_data.size();i++){
        wt[i]=to_data[i].get();
    }
    ta = &task_allocator;
}

// Move constructor: transfer ownership from another dispatcher
template<typename Gs_t, bool Singleblock, size_t Alpha>
ComputeDispatcher<Gs_t, Singleblock, Alpha>::ComputeDispatcher(ComputeDispatcher && rhs): group_size(rhs.group_size)
{
    gs = rhs.gs;  rhs.gs = nullptr;
    ta = rhs.ta;  rhs.ta = nullptr;
    rd = std::move(rhs.rd);
    wt = std::move(rhs.wt);
}

template<typename Gs_t, bool Singleblock, size_t Alpha>
// Wait for any task queue to be ready (empty) in given range
int ComputeDispatcher<Gs_t, Singleblock, Alpha>::wait_any_deployed(int begin_id, int end_id){
    while(true){
        for(int i = begin_id; i <= end_id;i++){
            if(ta->q[i].emptied()) return i;
        }
        std::this_thread::yield();
    }
}

template<typename Gs_t, bool Singleblock, size_t Alpha>
// Wait for any worker to finish in given range
int ComputeDispatcher<Gs_t, Singleblock, Alpha>::wait_any_finished(int begin_id, int end_id){
    while(true){
        for(int i = begin_id; i <= end_id;i++){
            if( ta->is_finished(i) ) return i;
        }
        // if constexpr(NUM_DEVICES>=3){
        //     LOG_DEBUG(0, "is 3 finished?: "<<ta->is_finished(3));
        //     if(!ta->is_finished(3)){
        //         LOG_DEBUG(0, "-dep="<<ta->num_deployed[3]<<"-fin="<<ta->num_finished[3]);
        //     }
        // }
        std::this_thread::yield();
    }
}

// Convert worker ID to readable string (CPU or GPU#)
static std::string getIdString(int id){
    if(id == 0) return "CPU";
    else{
        return "GPU"+std::to_string(id-1);
    }
}

template<typename Gs_t, bool Singleblock, size_t Alpha>
// Decide which worker (CPU or GPU) should handle a block based on flops
// Returns 0 for CPU, 1+ for GPU devices
int ComputeDispatcher<Gs_t, Singleblock, Alpha>::get_rb_type(unsigned long long flops){
    if(flops > gs->max_allowed_flop){
        return 0;
    }
    if(flops > gs->gpu_flop_thresh){
        return wait_any_finished(1, num_gpus); 
    }
    if(flops < Alpha * 32){
        return 0;
    }
    int slaveId = wait_any_finished(0, num_gpus);
    return slaveId;
}

template<typename Gs_t, bool Singleblock, size_t Alpha>
// Try to assign to GPU if memory allows, otherwise CPU
int ComputeDispatcher<Gs_t, Singleblock, Alpha>::get_rb_type_try_gpu(ull flops){
    if(flops > gs->max_allowed_flop){
      return 0;
    }
    int slaveId = wait_any_finished(1, num_gpus);
    return slaveId;
}

template<typename Gs_t, bool Singleblock, size_t Alpha>
// Dispatch a row block to appropriate worker (CPU or GPU)
// tid: row block ID, buffer: which output buffer to use
int ComputeDispatcher<Gs_t, Singleblock, Alpha>::work(int tid, int buffer){
    ull flop = gs->block_flops[tid];
    ComputeResultData<Gs_t, Singleblock> &res = *wt[buffer];
    // Reset block status for this row block
    res.setBlockStatusAll(0);
    // Notify all workers to prepare for new row block
    for(int i=0;i<num_workers;i++){
        ta->push_at(i, encode_row_buffer(tid, buffer), false); //change row
    }
    // Select worker based on flops
    int worker_id = get_rb_type(flop);
    // int worker_id = get_rb_type_try_gpu(flop);
    // LOG_DEBUG(1, "NUM_DEVICES "<<num_gpus<<" isfin?="<<ta->is_finished(3));
    LOG_DEBUG(1, "BLOCKTYPE "<<tid<<"="<<getIdString(worker_id));
    
    // CPU worker: dispatch entire row block
    if(worker_id == 0){
        ta->push_at(0/*workerid*/, 0/*cbid*/);
        profiler::Instance().cpu_data.blocks_cpu ++;
    }
    // GPU workers: dispatch column blocks in groups
    else{
        // Get read data for this buffer
        Rd_t &read_data = *rd.at(buffer);
        // Dispatch column blocks to GPUs in batches
        int blocks_count = 0;
        for(int idx: read_data.cBlockId){
            // Skip empty column blocks
            EIdType cb_nnz = read_data.cbNNZ[idx];
            if(cb_nnz == 0){
                res.setBlockStatus(idx,255);
                profiler::Instance().cpu_data.skipped_col_block += 1;
                continue;
            }
            ta->push_at(worker_id, idx);
            // Switch to next GPU after group_size blocks
            if(++blocks_count == group_size){
                blocks_count = 0;
                worker_id = wait_any_finished(1, num_gpus);
            }
            // printf("rbid=%d, push idx%d to GPU:\n", tid, idx);
        }
        profiler::Instance().cpu_data.blocks_gpu++;
    }
    for(int i=0;i<num_workers;i++){
        LOG_DEBUG(1, "After Deploy "<<i<<":("<<ta->num_deployed[i]<<","<<ta->num_finished[i]<<")");
    }
    return 0;
    // return ta->wait_all_deployed();
}



template struct ComputeDispatcher<default_gs_t, true, 16>;
template struct ComputeDispatcher<default_gs_t, false, 16>;
// template struct ComputeDispatcher<default_gs_t, true, 16>;
// template struct ComputeDispatcher<default_gs_t, false, 16>;




} //namespace CESpGEMM