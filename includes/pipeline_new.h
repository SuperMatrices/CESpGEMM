#pragma once
#include<queue>
#include<vector>
#include<atomic>
#include<mutex>
#include<condition_variable>
#include<optional>
#include<cassert>
#include<array>
#include"ThreadSafeQueue.h"
#include"ErrorNotifier.h"
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "logger.h"

namespace CESpGEMM{
namespace PIPE_LINE_1{

struct SymStage
{
  sem_t _avail;
  sem_t _empty;
  SymStage(){
    sem_init(&_avail, 0, 0);
    sem_init(&_empty, 0, 1);
  }
  void post(){
    SEM_POST(&_avail);  // Signal that data is available
  }
  void require(){
    SEM_WAIT(&_avail);  // Wait for data to be available
  }
  void wait_empty(){
    SEM_WAIT(&_empty);  // Wait for stage to become empty
  }
  void post_empty(){
    SEM_POST(&_empty);  // Signal that stage is now empty
  }
  void wait_avail(){
    SEM_WAIT(&_avail);  // Wait for data availability
  }
  void post_avail(){
    SEM_POST(&_avail);  // Signal data availability
  }
} ;


// Main pipeline execution function with three-stage parallel processing
// Gs_t: Global storage type
// PrepareW: Worker type for preparation stage
// ComputeW: Worker type for computation stage
// MergeW: Worker type for merging stage
template<typename Gs_t, typename PrepareW, typename ComputeW, typename MergeW>
void run_pipeline(
  Gs_t *gs,
  PrepareW &prepare_worker,
  ComputeW &compute_worker,
  MergeW &merge_worker,
  std::vector<IdxType> input_tasks,      // Input task IDs to process
  std::vector<IdxType> &finished_results, // Output: completed task IDs
  ErrorNotifier &notifier                 // Error notification mechanism
){
  // Thread-safe queues for passing tasks between stages
  ThreadSafeQueue<IdxType> q_prepare, q_compute, q_merge;
  std::array<SymStage, 2> prepare_stage;
  std::array<SymStage, 2> compute_stage;
  std::array<SymStage, 2> merge_stage;

  
  for(auto t: input_tasks) q_prepare.enque(t);

  // Prepare thread: fetches tasks and prepares data for computation
  auto prepare_thread = std::thread([&notifier, &prepare_worker, &prepare_stage, &compute_stage, &q_prepare, &q_compute](){
    printf("Prepare Worker (%lx, LWP %d)\n", pthread_self(), gettid());
    for(bool buffer=0;;buffer^=1){  // Toggle between 0 and 1 for double buffering
      if(notifier.has_error()){
        break;
      }
      IdxType tid;
      bool is_empty = q_prepare.empty_or_pop(tid);
      assert(tid >= 0||tid==-1);
      if(is_empty) break;
      
      // Wait for compute stage buffer to be empty before writing
      compute_stage[buffer].wait_empty();
      
      LOG_DEBUG(1, "PrepareWork Computing: "<<tid);
      int ret = prepare_worker.work(tid, buffer);  // Perform preparation work
      if(ret){
        notifier.notify();
        break;
      }
      
      // Signal that data is available and stage is done
      prepare_stage[buffer].post_avail();
      prepare_stage[buffer].post_empty();
      
      // Pass task to compute stage
      q_compute.enque(tid);
    }
    // Cleanup: signal all stages as empty and send termination signal
    prepare_stage[0].post_empty();
    prepare_stage[1].post_empty();
    q_compute.enque(-1);  // -1 signals termination
    prepare_worker.finish();
  } ) ;

  // Compute thread: performs the main computation on prepared data
  auto compute_thread = std::thread( [&notifier, &compute_worker, &compute_stage, &merge_stage, &q_compute, &q_merge](){
    printf("Compute Worker (%lx, LWP %d)\n", pthread_self(), gettid());
    for(bool buffer = 0; ; buffer^=1){  // Toggle between buffers
      if(notifier.has_error()) break;
      IdxType tid = q_compute.pop();
      assert(tid >= 0||tid==-1);
      if(tid==-1) break;  // Termination signal
      
      // Wait for merge stage buffer to be empty
      merge_stage[buffer].wait_empty();
      
      LOG_DEBUG(1, "ComputeWork Computing: "<<tid);
      int ret = compute_worker.work(tid, buffer);  // Perform computation
      if(ret){
        notifier.notify();
        break;
      }
      
      // Signal completion and availability
      compute_stage[buffer].post_avail();
      compute_stage[buffer].post_empty();
      
      // Pass task to merge stage
      q_merge.enque(tid);
    }
    // Cleanup: signal all stages as empty and send termination
    compute_stage[0].post_empty();
    compute_stage[1].post_empty();
    q_merge.enque(-1);
    compute_worker.finish();
  } ) ;

  // Merge thread: merges computation results and writes final output
  auto merge_thread = std::thread( [&notifier, &merge_worker, &merge_stage, &q_merge, &finished_results](){
    printf("Merge Worker (%lx, LWP %d)\n", pthread_self(), gettid());
    for(bool buffer = 0;;buffer^=1){  // Toggle between buffers
      if(notifier.has_error()) break;
      IdxType tid = q_merge.pop();
      assert(tid >= 0||tid==-1);
      if(tid==-1) break;  // Termination signal
      
      LOG_DEBUG(1, "MergeWork Computing: "<<tid);
      int ret = merge_worker.work(tid, buffer);  // Perform merge operation
      if(ret){
        notifier.notify();
        break;
      }
      
      // Signal completion and availability
      merge_stage[buffer].post_avail();
      merge_stage[buffer].post_empty();
      
      // Add completed task to results
      finished_results.push_back(tid);
    }
    // Cleanup: signal all stages as empty
    merge_stage[0].post_empty();
    merge_stage[1].post_empty();
    merge_worker.finish();
  } ) ;

  // Check for errors before joining threads
  if(notifier.has_error()){
    printf("WHAT : notifier has error!\n");
  }

  // Wait for all threads to complete
  prepare_thread.join();
  compute_thread.join();
  merge_thread.join();
  

  
}

}//ppl
}//cespgemm
