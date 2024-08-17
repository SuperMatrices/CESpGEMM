#include"profiler.h"
#include<cstdio>
#include"helper.cuh"
namespace CESpGEMM
{
CudaStreamTimer::CudaStreamTimer():avail{std::atomic_bool(0)}{
  cudaEventCreate(&start_point);
  cudaEventCreate(&end_point);
}

CudaStreamTimer::~CudaStreamTimer(){
  cudaEventDestroy(start_point);
  cudaEventDestroy(end_point); 
}
void CudaStreamTimer::record_start(cudaStream_t stream){
  CHK_CUDA_ERR(cudaEventRecord(start_point, stream));
}

void CudaStreamTimer::record_end(cudaStream_t stream){
  CHK_CUDA_ERR(cudaEventRecord(end_point, stream));
}

double CudaStreamTimer::get_time_ms(){
  float ret=0;
  CHK_CUDA_ERR(cudaEventElapsedTime(&ret, start_point, end_point));
  return ret;
}

double CudaStreamTimer::consume(){
  if(avail.load()==false) return 0.0;
  avail = false;
  return get_time_ms();
}


profiler* profiler::single = nullptr;

profiler& profiler::Instance(){
  return *single;
}
void profiler::Init(){
  single = new profiler{};
}
profiler::profiler():_epoch(std::chrono::steady_clock::now()){
}
profiler::~profiler(){
  printf("~profiler");
  
}
long long profiler::get_tick_since_start(){
  using namespace std::chrono;
  return duration_cast<microseconds>(steady_clock::now() - _epoch).count();
}  
} // namespace CESpGEMM
