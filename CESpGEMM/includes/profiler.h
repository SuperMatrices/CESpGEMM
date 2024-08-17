#pragma once
#include<cuda.h>
#include<cuda_runtime.h>
#include<chrono>
#include<atomic>
#include<map>
#include<vector>

namespace CESpGEMM
{
struct CudaStreamTimer
{
  cudaEvent_t start_point;
  cudaEvent_t end_point;
  std::atomic_bool avail;
  CudaStreamTimer();

  ~CudaStreamTimer();

  void record_start(cudaStream_t s);
  void record_end(cudaStream_t s);
  double consume();
  double get_time_ms();
} ;

class profiler
{
public:
  static profiler& Instance();
  static void Init();
  std::chrono::_V2::steady_clock::time_point _epoch;
  double convert_vcsr_time=0;
  double compress_time=0;
  double compress_ratio=0;
  long long compressd_len=0;
  double throughput_kBpS=0;
  double cpu_compute_time=0;
  double merge_time=0;
  double prepare_time=0;
  double io_time=0;
  double h2d_time=0;
  double kernel_time=0;
  double d2h_time=0;
  double gpu_computer_time=0;
  double t_pipeline=0;
  int skipped_col_block=0;
  int blocks_cpu=0, blocks_gpu=0;
  double kbytes_h2d=0;
  double kbytes_d2h=0;
  double kbytes_io=0;
  CudaStreamTimer timer_h2d;
  CudaStreamTimer timer_knl;
  CudaStreamTimer timer_d2h;
  std::map<uintptr_t, std::vector<std::pair<long long,long long>>> worker_trace;
  long long get_tick_since_start();

private:
  profiler();
  ~profiler();
  static profiler* single;

} ;

template<typename Tm> 
double get_chrono_ms(Tm t_start, Tm t_end){
  return std::chrono::duration_cast<std::chrono::microseconds>(t_end-t_start).count()/1000.0;
}



} // namespace CESpGEMM
