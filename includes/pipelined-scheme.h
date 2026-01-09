#pragma once
#include<iostream>
#include<cstdio>
#include"CSR.h"
#include"task_alloc.h"
#include"pipeline.h"
#include"pipeline_new.h"
#include"Storage.h"
#include "WritableStorage.h"
// #include "pipeline-worker-impl.h"
#include"prepare.h"
#include"compute-dispatch.h"
#include"merge.h"
// #include"cpu_task_fetcher.h"
#include"cpu_task_fetcher_row.h"
#include"gpu_task_fetcher_async.h"
#include"gpu_task_fetcher_mgpu.h"
#include"avail_devices.h"

// #include"gpu_fetcher.h"

namespace CESpGEMM{
/*
namespace PIPE_LINE{

template<typename SrcEIdType, typename EIdType, typename ValType, bool SingleBlock>
void do_works(GlobalStorage<SrcEIdType, EIdType, ValType>*gs, bool file_write, std::string result_path, bool sampling, std::vector<IdxType> sampled_tasks, int device_id){
  using std::vector;
  using Gs_t = GlobalStorage<SrcEIdType, EIdType, ValType>;
  vector<NullStage> empty_stage(2);
  vector<SymStage> prepare_stage(2);
  vector<SymStage> compute_stage(2);
  // vector<SymStage> merge_stage(2);
  using task_buffer_t = StageTaskBuffer<IdxType, 2>;
  // task_buffer_t prepare_task_buffer;
  // vector<EIdType> result_block_nnz(gs->)
  task_buffer_t compute_task_buffer;
  task_buffer_t merge_task_buffer;
  std::vector<IdxType> input_task_buffer;
  if(!sampling){
    for(int i=0;i<gs->numBlocksA;i++){ input_task_buffer.push_back(i); }
  }
  else{
    input_task_buffer.swap(sampled_tasks);
  }
  int num_tasks_input = input_task_buffer.size();
  input_task_buffer.push_back(-1);

  OneSideBuffer<IdxType> prepare_task_buffer(std::move(input_task_buffer));
  OneSideBuffer<IdxType> empty_final_task_buffer{};
  // EmptyTaskBuffer<int> empty_task_buffer;

  vector<FlopData<EIdType>> flop_info(2, FlopData<EIdType>(gs->numBlocksB));
  
  vector<SrcEIdType> block_nnzc_count (gs->numBlocksA);
  vector<MergedResultData<SrcEIdType, ValType>> merge_result;
  merge_result.reserve(2);
  merge_result.emplace_back(gs->pool_size, block_nnzc_count.data());
  merge_result.emplace_back(gs->pool_size, block_nnzc_count.data());

  using prepare_worker_t = PrepareWorker<Gs_t>;
  using compute_worker_t = ComputeDispatcher<Gs_t, SingleBlock, 16>;
  using merge_worker_t = MergeWorker<Gs_t, SingleBlock>;

  prepare_worker_t prepare_worker(gs, std::ref(flop_info));
  TaskAlloc<2, int> compute_tasks_allocator{};

  vector< std::unique_ptr<ComputeResultData<Gs_t, SingleBlock> > > compute_result;
  compute_result.reserve(2);
  compute_result.push_back( std::make_unique<ComputeResultData<Gs_t, SingleBlock> >(gs->numBlocksB, gs->blockSizeA) );
  compute_result.push_back( std::make_unique<ComputeResultData<Gs_t, SingleBlock> >(gs->numBlocksB, gs->blockSizeA) );
  compute_worker_t compute_worker(gs, std::ref(flop_info), std::ref(compute_result), compute_tasks_allocator);
  
  
  // CPU_TaskFetcher<Gs_t, SingleBlock> cpu_fetcher(
  //   std::ref(compute_tasks_allocator),
  //   gs,
  //   std::ref(compute_result)
  // );

  MKL_computer<Gs_t, SingleBlock> mkl_computer (
    gs->num_workers, gs, std::ref(compute_result)
  ) ;

  const csc<SrcEIdType, ValType> &csrb_t_ = *gs->csrB_T.get(); 
  const csr<SrcEIdType, ValType> &csra_ = *gs->csrA.get();
  size_t max_block_nnzA = 0, max_block_nnzB = 0;
  {
    int nbb = gs->numBlocksB;
    for(int i=0;i<nbb;i++){
      int cStart = i * gs->blockSizeB, cEnd = std::min<int>(cStart+gs->blockSizeB, csrb_t_.nr);
      int nnz = csrb_t_.ptr[cEnd]-csrb_t_.ptr[cStart];
      max_block_nnzB = std::max<ull>(max_block_nnzB, nnz);
    }
    
    for(IdxType i=0,L=0;i<gs->numBlocksA;i++,L+=gs->blockSizeA){
      IdxType R = std::min(L + gs->blockSizeA, csra_.nr);
      max_block_nnzA = std::max<ull>(max_block_nnzA, csra_.ptr[R]-csra_.ptr[L]);
    }
  }
  
  // printf("maxnnzA=%lld  maxnnzB=%lld!\n", max_block_nnzA, max_block_nnzB);
  // GpuComputer<EIdType, ValType> gpu_computer(
  //   gs->blockSizeA, gs->blockSizeB, 
  //   max_block_nnzA, max_block_nnzB, 
  //   0, 
  //   gs, 
  //   1ll<<20,
  //   std::ref(compute_result)
  // );

  // printf("gs->max_rcblockflop=%lld", gs->max_rcblock_flop);
  // printf("blocksizeB=%d\n", gs->blockSizeB);
  GpuComputerAsync<Gs_t, SingleBlock> gpu_computer_async(
    gs->blockSizeA, gs->blockSizeB, max_block_nnzA, max_block_nnzB,
    device_id,
    gs, std::ref(compute_result), profiler::Instance().gpu_data.get()
  ) ;

  ull max_flop = std::min((ull)gs->max_allowed_flop, gs->max_rcblock_flop);
  printf("trying to initialize....\n");
  if(!gpu_computer_async.try_initialize(max_block_nnzA, max_block_nnzB, max_flop)){
    printf("initialize failed\n");
    throw std::exception{}; 
  }

  CPU_TaskFetcher_by_row<Gs_t, SingleBlock>  cpu_fetcher_row(
    std::ref(compute_tasks_allocator),
    gs, 
    std::ref(mkl_computer)
  ) ;


  GPUTaskFetcherAsync<Gs_t, SingleBlock> gpu_fetcher(
    std::ref(compute_tasks_allocator),
    std::ref(gpu_computer_async), 
    std::ref(flop_info)
  ) ;

  Map_Util &mp_inst = Map_Util::Instance();
  merge_worker_t merge_worker(
    gs,
    std::ref(compute_result),
    std::ref(merge_result),
    result_path
  ) ; 

  auto t_start_pipeline = std::chrono::steady_clock::now();

  PipeLineWorker<NullStage, SymStage, 
  OneSideBuffer<IdxType>, task_buffer_t, 
  prepare_worker_t> ppl_prepare(
    std::move(prepare_worker), 
    std::ref(empty_stage),
    std::ref(prepare_stage),
    std::ref(prepare_task_buffer),
    std::ref(compute_task_buffer)
  ) ;
  mp_inst.add(&ppl_prepare, "ppl_prepare");
  PipeLineWorker<SymStage, SymStage, 
  task_buffer_t, task_buffer_t, 
  compute_worker_t> ppl_compute(
    std::move(compute_worker),
    std::ref(prepare_stage),
    std::ref(compute_stage),
    std::ref(compute_task_buffer),
    std::ref(merge_task_buffer)
  ) ;
  mp_inst.add(&ppl_compute, "ppl_compute");

  PipeLineWorker<SymStage, NullStage, 
  task_buffer_t, OneSideBuffer<IdxType>, 
  merge_worker_t> ppl_merge(
    std::move(merge_worker),
    std::ref(compute_stage),
    std::ref(empty_stage),
    std::ref(merge_task_buffer),
    std::ref(empty_final_task_buffer)
  ) ;
  mp_inst.add(&ppl_merge, "ppl_merge");

////////////TEST/////////////////
  // prepare_task_buffer.post(0);
  // prepare_task_buffer.post(-1);
  // return;
////////////TEST/////////////////
  
  vector<IdxType> &finished_result=empty_final_task_buffer.tasks;
  while(finished_result.size()<num_tasks_input){
    using namespace std::chrono_literals;
    // printf("! finished result sz=%d\n", finished_result.size());
    std::this_thread::sleep_for(5ms);
  }

  auto t_end_pipeline = std::chrono::steady_clock::now();
  profiler::Instance().cpu_data.t_pipeline += get_chrono_ms(t_start_pipeline, t_end_pipeline);


  EIdType total_nnz = 0;
  // printf("total_nnz collecting final tasks!:\n");
  for(int i:finished_result){
    // printf("empty final task buffer: %d, finished!\n",i);
    if(i!=-1){
      total_nnz += block_nnzc_count[i];
      // printf("%d blocknnz=%lld\n", i, block_nnzc_count[i]);
    }
  }
  printf("total_nnz = %lld\n", total_nnz);
  // if(total_nnz == 0){
  //   printf("blocks computed when nnz==0: %d, num_tasks_input=%d, blocksizeA=%d\n", finished_result.size(), num_tasks_input, gs->blockSizeA);
  //   // throw;
  // }
  // else{
  //   printf("blocks computed when nnz>0: %d, num_tasks_input=%d, blocksizeA=%d\n", finished_result.size(), num_tasks_input, gs->blockSizeA);
  // }

  // for(int i=0,buffer=0;i<gs->numBlocksA;i++,buffer^=1){
  //   SymStage&last_stage = merge_stage[buffer];
  //   printf("main thread require result on [%d,%d]\n", i, buffer);
  //   last_stage.require();
  //   printf("main thread post_empty on [%d,%d]\n", i, buffer);
  //   last_stage.post_empty();
  // }
  
}

}//PIPE_LINE

namespace PIPE_LINE_1{
template<typename SrcEIdType, typename EIdType, typename ValType, bool SingleBlock>
double do_works_new(GlobalStorage<SrcEIdType, EIdType, ValType>*gs, bool file_write, std::string result_path, bool sampling, std::vector<IdxType> sampled_tasks, int device_id){
  using std::vector;
  using Gs_t = GlobalStorage<SrcEIdType, EIdType, ValType>;
  std::vector<IdxType> input_task_buffer;
  if(!sampling){
    for(int i=0;i<gs->numBlocksA;i++){ input_task_buffer.push_back(i); }
  }
  else{
    input_task_buffer.swap(sampled_tasks);
  }
  int num_tasks_input = input_task_buffer.size();

  vector<FlopData<EIdType>> flop_info(2, FlopData<EIdType>(gs->numBlocksB));  
  vector<SrcEIdType> block_nnzc_count (gs->numBlocksA);
  vector<MergedResultData<SrcEIdType, ValType>> merge_result;
  merge_result.reserve(2);
  merge_result.emplace_back(gs->pool_size, block_nnzc_count.data());
  merge_result.emplace_back(gs->pool_size, block_nnzc_count.data());

  using prepare_worker_t = PrepareWorker<Gs_t>;
  using compute_worker_t = ComputeDispatcher<Gs_t, SingleBlock, 16>;
  using merge_worker_t = MergeWorker<Gs_t, SingleBlock>;

  prepare_worker_t prepare_worker(gs, std::ref(flop_info));
  TaskAlloc<2, int> compute_tasks_allocator{};

  vector< std::unique_ptr<ComputeResultData<Gs_t, SingleBlock> > > compute_result;
  compute_result.reserve(2);
  compute_result.push_back( std::make_unique<ComputeResultData<Gs_t, SingleBlock> >(gs->numBlocksB, gs->blockSizeA) );
  compute_result.push_back( std::make_unique<ComputeResultData<Gs_t, SingleBlock> >(gs->numBlocksB, gs->blockSizeA) );
  compute_worker_t compute_worker(gs, std::ref(flop_info), std::ref(compute_result), compute_tasks_allocator);

  MKL_computer<Gs_t, SingleBlock> mkl_computer (
    gs->num_workers, gs, std::ref(compute_result)
  ) ;

  const csc<SrcEIdType, ValType> &csrb_t_ = *gs->csrB_T.get(); 
  const csr<SrcEIdType, ValType> &csra_ = *gs->csrA.get();
  size_t max_block_nnzA = 0, max_block_nnzB = 0;
  {
    int nbb = gs->numBlocksB;
    for(int i=0;i<nbb;i++){
      int cStart = i * gs->blockSizeB, cEnd = std::min<int>(cStart+gs->blockSizeB, csrb_t_.nr);
      int nnz = csrb_t_.ptr[cEnd]-csrb_t_.ptr[cStart];
      max_block_nnzB = std::max<ull>(max_block_nnzB, nnz);
    }
    
    for(IdxType i=0,L=0;i<gs->numBlocksA;i++,L+=gs->blockSizeA){
      IdxType R = std::min(L + gs->blockSizeA, csra_.nr);
      max_block_nnzA = std::max<ull>(max_block_nnzA, csra_.ptr[R]-csra_.ptr[L]);
    }
  }
  
  using GpuComputer_t = GpuComputerAsync<Gs_t, SingleBlock> ;
  
  std::vector<std::unique_ptr<GpuComputer_t> > gpu_computer_async_ptrs;
  gpu_computer_async_ptrs.reserve(1);
  gpu_computer_async_ptrs.push_back(
    std::move(
      std::make_unique<GpuComputer_t>(
        gs->blockSizeA, gs->blockSizeB, max_block_nnzA, max_block_nnzB,
        device_id,
        gs, std::ref(compute_result), profiler::Instance().gpu_data.get()
      )
    )
  );


  ull max_flop = std::min((ull)gs->max_allowed_flop, gs->max_rcblock_flop);
  printf("trying to initialize....\n");
  if(!gpu_computer_async_ptrs[0]->try_initialize(max_block_nnzA, max_block_nnzB, max_flop)){
    printf("initialize failed\n");
    return 1e100;
  }

  CPU_TaskFetcher_by_row<Gs_t, SingleBlock>  cpu_fetcher_row(
    std::ref(compute_tasks_allocator),
    gs, 
    std::ref(mkl_computer)
  ) ;


  GPUTaskFetcherAsync<Gs_t, SingleBlock> gpu_fetcher(
    std::ref(compute_tasks_allocator),
    std::ref(*gpu_computer_async_ptrs[0].get()), 
    std::ref(flop_info)
  ) ;

  // GPUTaskFetcherAsyncMGPU<Gs_t, SingleBlock> gpu_fetcher_mgpu(
  //   std::ref(compute_tasks_allocator),
  //   std::ref(gpu_computer_async_ptrs),
  //   std::ref(flop_info)
  // ) ;

  Map_Util &mp_inst = Map_Util::Instance();
  merge_worker_t merge_worker(
    gs,
    std::ref(compute_result),
    std::ref(merge_result),
    result_path
  ) ; 

  auto t_start_pipeline = std::chrono::steady_clock::now();

  std::vector<IdxType> finished_tasks;
  ErrorNotifier notifier;

  run_pipeline(gs, prepare_worker, compute_worker, merge_worker, input_task_buffer, finished_tasks, notifier);
  

////////////TEST/////////////////
  // prepare_task_buffer.post(0);
  // prepare_task_buffer.post(-1);
  // return;
////////////TEST/////////////////
  
  auto t_end_pipeline = std::chrono::steady_clock::now();
  profiler::Instance().cpu_data.t_pipeline += get_chrono_ms(t_start_pipeline, t_end_pipeline);

  if(notifier.has_error()){
    return 1e100;
  }

  EIdType total_nnz = 0;
  for(int i:finished_tasks){
    total_nnz += block_nnzc_count[i];
    // printf("%d blocknnz=%d\n", i, block_nnzc_count[i]);
  }
  printf("total_nnz = %lld\n", (long long)total_nnz);
  return profiler::Instance().cpu_data.t_pipeline;
}

}//PIPE_LINE_1
*/


// Multi-GPU pipeline execution function
// Performs sparse matrix-matrix multiplication (SpGEMM) using multiple GPUs
namespace PIPE_LINE_1{

template<typename SrcEIdType, typename EIdType, typename ValType, bool SingleBlock>
double do_works_mgpu(GlobalStorage<SrcEIdType, EIdType, ValType>*gs, bool file_write, std::string result_path, bool sampling, std::vector<IdxType> sampled_tasks){
  constexpr int num_devices = NUM_DEVICES;
  using std::vector;
  using Gs_t = GlobalStorage<SrcEIdType, EIdType, ValType>;
  
  // Prepare input tasks: either all blocks or sampled subset
  vector<IdxType> input_task_buffer;
  if(!sampling){
    for(int i=0;i<gs->numBlocksA;i++) input_task_buffer.push_back(i);
  }
  else{
    input_task_buffer.swap(sampled_tasks);
  }
  int num_tasks_input = input_task_buffer.size();
  
  // Double-buffered data structures for pipeline stages
  vector<FlopData<EIdType>> flop_info(2, FlopData<EIdType>(gs->numBlocksB));  // Flop count info
  vector<SrcEIdType> block_nnzc_count(gs->numBlocksA);  // Non-zero count per block
  vector<MergedResultData<SrcEIdType, ValType>> merge_result;  // Merged results buffer
  merge_result.reserve(2);
  merge_result.emplace_back(gs->pool_size, block_nnzc_count.data());
  merge_result.emplace_back(gs->pool_size, block_nnzc_count.data());
  
  // Worker types for each pipeline stage
  using prepare_worker_t = PrepareWorker<Gs_t>;
  using compute_worker_t = ComputeDispatcher<Gs_t, SingleBlock, 16>;
  using merge_worker_t = MergeWorker<Gs_t, SingleBlock>;
  
  // Initialize workers
  prepare_worker_t prepare_worker(gs, std::ref(flop_info));
  TaskAlloc<num_devices+1, int> compute_task_allocator;  // Task allocator for GPUs + CPU
  
  // Double-buffered compute result storage
  std::vector< std::unique_ptr<ComputeResultData<Gs_t, SingleBlock> > > compute_result;
  compute_result.reserve(2);
  compute_result.push_back( std::make_unique<ComputeResultData<Gs_t, SingleBlock> >(gs->numBlocksB, gs->blockSizeA) );
  compute_result.push_back( std::make_unique<ComputeResultData<Gs_t, SingleBlock> >(gs->numBlocksB, gs->blockSizeA) );
  
  constexpr int group_size = 8;
  compute_worker_t compute_worker(gs, std::ref(flop_info), std::ref(compute_result), compute_task_allocator, group_size);
  // MKL-based CPU fallback computer
  MKL_computer<Gs_t, SingleBlock> mkl_computer(
    gs->num_workers, gs, std::ref(compute_result)
  ) ;

  // Calculate maximum non-zero counts per block for memory allocation
  const csc<SrcEIdType, ValType> &csrb_t_ = *gs->csrB_T.get();
  const csr<SrcEIdType, ValType> &csra_ = *gs->csrA.get();
  size_t max_block_nnzA = 0, max_block_nnzB = 0;
  {
    // Find max nnz in B blocks
    int nbb = gs->numBlocksB;
    for(int i=0;i<nbb;i++){
      int cStart = i * gs->blockSizeB, cEnd = std::min<int>(cStart+gs->blockSizeB, csrb_t_.nr);
      int nnz = csrb_t_.ptr[cEnd]-csrb_t_.ptr[cStart];
      max_block_nnzB = std::max<ull>(max_block_nnzB, nnz);
    }
    
    // Find max nnz in A blocks
    for(IdxType i=0,L=0;i<gs->numBlocksA;i++,L+=gs->blockSizeA){
      IdxType R = std::min(L + gs->blockSizeA, csra_.nr);
      max_block_nnzA = std::max<ull>(max_block_nnzA, csra_.ptr[R]-csra_.ptr[L]);
    }
  }
  // Initialize GPU computers for each device
  using GpuComputer_t = GpuComputerAsync<Gs_t, SingleBlock>;
  std::vector<std::unique_ptr<GpuComputer_t>> gpu_computers_async;
  gpu_computers_async.reserve(num_devices);
  auto devices = GPUSelection::get_device_map();
  
  // Create one GPU computer per device
  for(int i=0;i<num_devices;i++){
    gpu_computers_async.push_back(std::make_unique<GpuComputer_t>(
      gs->blockSizeA, gs->blockSizeB, max_block_nnzA, max_block_nnzB,
      devices[i], gs, std::ref(compute_result), profiler_mgpu::Instance().gpu_data[i].get()
    ) );
  }
  
  // Initialize all GPU computers with memory and flop limits
  ull max_flop = std::min((ull)gs->max_allowed_flop, gs->max_rcblock_flop);
  printf("trying to initialize...\n");
  for(auto &gcomp_ptr : gpu_computers_async){
    if(!gcomp_ptr->try_initialize(max_block_nnzA, max_block_nnzB, max_flop)){
      printf("initialize failed\n");
      return 1e100;
    }
  }

  // Task fetchers: CPU (MKL) and GPU
  CPU_TaskFetcher_by_row<Gs_t, SingleBlock> cpu_fetcher_row(
    std::ref(compute_task_allocator),
    gs,
    std::ref(mkl_computer)
  ) ;

  GPUTaskFetcherAsyncMGPU<Gs_t, SingleBlock> gpu_fetcher(
    std::ref(compute_task_allocator),
    std::ref(gpu_computers_async),
    std::ref(flop_info)
  ) ;
  
  // Initialize merge worker for writing results
  Map_Util &mp_inst = Map_Util::Instance();
  merge_worker_t merge_worker(
    gs,
    std::ref(compute_result),
    std::ref(merge_result),
    result_path
  ) ;
  
  // Run the three-stage pipeline and measure time
  auto t_start_pipeline = std::chrono::steady_clock::now();
  std::vector<IdxType> finished_tasks;
  ErrorNotifier notifier;
  run_pipeline(gs, prepare_worker, compute_worker, merge_worker, input_task_buffer, finished_tasks, notifier);
  
  auto t_end_pipeline = std::chrono::steady_clock::now();
  profiler::Instance().cpu_data.t_pipeline += get_chrono_ms(t_start_pipeline, t_end_pipeline);
  
  // Check for errors and compute total non-zeros
  if(notifier.has_error()) return 1e100;
  EIdType total_nnz = 0;
  for(int i:finished_tasks) total_nnz += block_nnzc_count[i];
  printf("total_nnz = %lld\n", (long long)total_nnz);
  return profiler::Instance().cpu_data.t_pipeline;
} 

} //PIPE_LINE_1

}