#pragma once
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include<atomic>
#include<map>
#include<vector>
#include<mutex>
#include<memory>

namespace CESpGEMM
{
struct CudaStreamTimer
{
  int dev_id;
  cudaEvent_t start_point;
  cudaEvent_t end_point;
  std::atomic_bool avail;
  CudaStreamTimer(int device_id);
  ~CudaStreamTimer();

  void record_start(cudaStream_t s);
  void record_end(cudaStream_t s);
  double consume();
  double get_time_ms();
} ;

struct cpu_profile_t
{
  double convert_vcsr_time=0;
  double compress_time=0;
  double compress_ratio=0;
  long long compressd_len=0;
  double throughput_kBpS=0;
  double cpu_compute_time=0;
  double merge_time=0;
  double prepare_time=0;
  double io_time=0;
  double t_pipeline=0;
  int skipped_col_block=0;
  int blocks_cpu=0, blocks_gpu=0;
  double kbytes_io=0;
  double sorting_time=0;
  cpu_profile_t()=default;
} ;

struct gpu_profile_t
{
  double h2d_time=0;
  double kernel_time=0;
  double d2h_time=0;
  double gpu_computer_time=0;
  double kbytes_h2d=0;
  double kbytes_d2h=0;
  int dev_id;
  CudaStreamTimer timer_h2d;
  CudaStreamTimer timer_knl;
  CudaStreamTimer timer_d2h;
  gpu_profile_t(int device_id):dev_id(device_id), timer_h2d(device_id), timer_knl(device_id), timer_d2h(device_id){
    
  }
  ~gpu_profile_t()=default;
  gpu_profile_t(const gpu_profile_t&) = delete;
  gpu_profile_t& operator=(const gpu_profile_t&) = delete;

  gpu_profile_t(gpu_profile_t&&) noexcept = delete;
  gpu_profile_t& operator=(gpu_profile_t&&) noexcept = delete;
} ;

class profiler
{
public:
  static profiler& Instance();
  static void Init(int dev_id);
  static void reset();
  std::chrono::_V2::steady_clock::time_point _starting_epoch;
  int dev_id;
  cpu_profile_t cpu_data;
  std::unique_ptr<gpu_profile_t> gpu_data;
  std::map<uintptr_t, std::vector<std::pair<long long, long long>>> worker_trace;
  long long get_tick_since_start();

private:
  profiler(int dev_id);
  ~profiler();
  static profiler* single;
} ;

class profiler_mgpu{
public:
  static profiler_mgpu& Instance();
  static void Init(int dev_id);
  static void reset();
  std::chrono::_V2::steady_clock::time_point _starting_epoch;
  int num_devices;
  cpu_profile_t cpu_data;
  std::vector<std::unique_ptr<gpu_profile_t>> gpu_data;
  std::map<uintptr_t, std::vector<std::pair<long long,long long>>> worker_trace;
  long long get_tick_since_start();

private:
  profiler_mgpu(int num_gpus);
  ~profiler_mgpu();
  static profiler_mgpu* single;
} ;

template<typename Tm> 
double get_chrono_ms(Tm t_start, Tm t_end){
  return std::chrono::duration_cast<std::chrono::microseconds>(t_end-t_start).count()/1000.0;
}



} // namespace CESpGEMM
