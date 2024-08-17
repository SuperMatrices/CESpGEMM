#pragma once
#include<iostream>
#include<cstdio>
#include"CSR.h"
#include"task_alloc.h"
#include"pipeline.h"
#include"Storage.h"
#include "WritableStorage.h"
// #include "pipeline-worker-impl.h"
#include"prepare.h"
#include"compute-dispatch.h"
#include"merge.h"
// #include"cpu_task_fetcher.h"
#include"cpu_task_fetcher_row.h"
#include"gpu_task_fetcher_async.h"

namespace CESpGEMM{

namespace PIPE_LINE{
template<typename SrcEIdType, typename EIdType, typename ValType, bool SingleBlock>
void do_works(GlobalStorage<SrcEIdType, EIdType, ValType>*gs, bool file_write, std::string result_path){
  using std::vector;
  using Gs_t = GlobalStorage<SrcEIdType, EIdType, ValType>;
  vector<NullStage> empty_stage(2);
  vector<SymStage> prepare_stage(2);
  vector<SymStage> compute_stage(2);
  // vector<SymStage> merge_stage(2);
  using task_buffer_t = StageTaskBuffer<int, 2>;
  // task_buffer_t prepare_task_buffer;
  // vector<EIdType> result_block_nnz(gs->)
  task_buffer_t compute_task_buffer;
  task_buffer_t merge_task_buffer;
  std::vector<int> input_task_buffer(gs->numBlocksA+1);

  for(int i=0;i<gs->numBlocksA;i++){ input_task_buffer[i]= i; }
  input_task_buffer[gs->numBlocksA]=-1;
  OneSideBuffer<int> prepare_task_buffer(std::move(input_task_buffer));
  OneSideBuffer<int> empty_final_task_buffer{};
  // EmptyTaskBuffer<int> empty_task_buffer;

  vector<FlopData<EIdType>> flop_info(2, FlopData<EIdType>(gs->numBlocksB));
  
  vector<SrcEIdType> block_nnzc_count (gs->numBlocksA);
  vector<MergedResultData<SrcEIdType, ValType>> merge_result;
  merge_result.reserve(2);
  merge_result.emplace_back(gs->pool_size, block_nnzc_count.data());
  merge_result.emplace_back(gs->pool_size, block_nnzc_count.data());

  using prepare_worker_t = PrepareWorker<Gs_t>;
  using compute_worker_t = ComputeDispatcher<Gs_t, SingleBlock, 16, 30>;
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
  CPU_TaskFetcher_by_row<Gs_t, SingleBlock>  cpu_fetcher_row(
    std::ref(compute_tasks_allocator),
    gs, 
    std::ref(mkl_computer)
  ) ;



  size_t max_block_nnzA = 0, max_block_nnzB = 0;
  {
    int nbb = gs->numBlocksB;
    for(int i=0;i<nbb;i++){
      int cStart = i * gs->blockSizeB, cEnd = std::min<int>(cStart+gs->blockSizeB, gs->csrB_T.nr);
      int nnz = gs->csrB_T.ptr[cEnd]-gs->csrB_T.ptr[cStart];
      max_block_nnzB = std::max<ull>(max_block_nnzB, nnz);
    }
    auto &csrA = gs->csrA;
    for(IdxType i=0,L=0;i<gs->numBlocksA;i++,L+=gs->blockSizeA){
      IdxType R = std::min(L + gs->blockSizeA, csrA.nr);
      max_block_nnzA = std::max<ull>(max_block_nnzA, csrA.ptr[R]-csrA.ptr[L]);
    }
  }
  
  printf("maxnnzA=%lld  maxnnzB=%lld!\n", max_block_nnzA, max_block_nnzB);

  printf("gs->max_rcblockflop=%lld", gs->max_rcblock_flop);
  printf("blocksizeB=%d\n", gs->blockSizeB);
  GpuComputerAsync<Gs_t, SingleBlock> gpu_computer_async(
    gs->blockSizeA, gs->blockSizeB, max_block_nnzA, max_block_nnzB, 
    0,
    gs, gs->max_rcblock_flop, std::ref(compute_result)
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
  OneSideBuffer<int>, task_buffer_t, 
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
  task_buffer_t, OneSideBuffer<int>, 
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
  
  vector<int> &finished_result=empty_final_task_buffer.tasks;
  while(finished_result.size()<gs->numBlocksA){
    using namespace std::chrono_literals;
    // printf("! finished result sz=%d\n", finished_result.size());
    std::this_thread::sleep_for(5ms);
  }

  auto t_end_pipeline = std::chrono::steady_clock::now();
  profiler::Instance().t_pipeline += get_chrono_ms(t_start_pipeline, t_end_pipeline);


  EIdType total_nnz = 0;
  for(int i:finished_result){
    printf("empty final task buffer: %d, finished!\n",i);
    if(i!=-1){
      total_nnz += block_nnzc_count[i];
      printf("%d blocknnz=%lld\n", i, block_nnzc_count[i]);
    }
  }
  printf("total_nnz = %lld\n", total_nnz);

  // for(int i=0,buffer=0;i<gs->numBlocksA;i++,buffer^=1){
  //   SymStage&last_stage = merge_stage[buffer];
  //   printf("main thread require result on [%d,%d]\n", i, buffer);
  //   last_stage.require();
  //   printf("main thread post_empty on [%d,%d]\n", i, buffer);
  //   last_stage.post_empty();
  // }
  
}

}//PIPE_LINE

}