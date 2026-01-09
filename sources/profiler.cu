#include"profiler.h"
#include<cstdio>
#include"helper.cuh"
#include<assert.h>
#include"logger.h"
#include"avail_devices.h"
#include<nvml.h>

namespace CESpGEMM
{
// Constructor: Create CUDA events for timing
CudaStreamTimer::CudaStreamTimer(int device_id):avail{0}, dev_id(device_id){
  cudaSetDevice(dev_id);
  cudaEventCreate(&start_point);
  cudaEventCreate(&end_point);
}

// Destructor: Destroy CUDA events
CudaStreamTimer::~CudaStreamTimer(){
  cudaEventDestroy(start_point);
  cudaEventDestroy(end_point); 
}
// Record start event on stream
void CudaStreamTimer::record_start(cudaStream_t stream){
  int current_device = -1;
  cudaGetDevice(&current_device);
  CHK_CUDA_ERR(cudaEventRecord(start_point, stream));
}

// Record end event on stream
void CudaStreamTimer::record_end(cudaStream_t stream){
  CHK_CUDA_ERR(cudaEventRecord(end_point, stream));
}

// Get elapsed time between start and end events (milliseconds)
double CudaStreamTimer::get_time_ms(){
  float ret=0;
  CHK_CUDA_ERR(cudaEventElapsedTime(&ret, start_point, end_point));
  return ret;
}

// Consume and reset timer, return elapsed time
double CudaStreamTimer::consume(){
  if(avail.load()==false) return 0.0;
  avail = false;
  return get_time_ms();
}

profiler* profiler::single = nullptr;
profiler_mgpu* profiler_mgpu::single = nullptr;

// Get singleton profiler instance
profiler& profiler::Instance(){
  assert(single);
  return *single;
}
// Initialize profiler singleton with device ID
void profiler::Init(int dev_id){
  cudaSetDevice(dev_id);
  single = new profiler{dev_id};
}
// Reset profiler statistics
void profiler::reset(){
  assert(single!=nullptr);
  auto &s=*single;
  s._starting_epoch = std::chrono::steady_clock::now();
  s.cpu_data = cpu_profile_t{};
  s.gpu_data = std::make_unique<gpu_profile_t>(s.dev_id);
  s.worker_trace.clear();
}
// Constructor: Initialize with device and start time
profiler::profiler(int dev_id):_starting_epoch(std::chrono::steady_clock::now()), cpu_data(), dev_id(dev_id){
  cudaSetDevice(dev_id);
  gpu_data = std::make_unique<gpu_profile_t>(dev_id);
}
profiler::~profiler(){
}
// Get elapsed time since initialization (microseconds)
long long profiler::get_tick_since_start(){
  using namespace std::chrono;
  return duration_cast<microseconds>(steady_clock::now() - _starting_epoch).count();
}  


// Constructor: Initialize multi-GPU profiler
profiler_mgpu::profiler_mgpu(int num_gpus){
  num_devices = num_gpus;
  // Reserve space for per-GPU data
  gpu_data.reserve(num_devices);
  // Get list of available GPUs
  std::vector<int> devices = GPUSelection::get_device_map();
  // Print GPU information
  for(int dev: devices){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    std::cout << "Device " << dev << ": " << prop.name << std::endl;
    std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB\n";
    std::cout << "  MultiProcessorCount: " << prop.multiProcessorCount << "\n";
  }
  CHK_ASSERT(num_devices == (int)devices.size()) ;
  for(int i=0;i<num_devices;i++){
    gpu_data.push_back(std::make_unique<gpu_profile_t>(devices[i]));
  }
}

// Initialize multi-GPU profiler singleton
void profiler_mgpu::Init(int num_gpus){
  single = new profiler_mgpu(num_gpus);
}

// Reset multi-GPU profiler statistics
void profiler_mgpu::reset(){
  assert(single!=nullptr);
  auto &s = *single;
  s.cpu_data = cpu_profile_t{};
  s._starting_epoch = std::chrono::steady_clock::now();
  std::vector<int> devices = GPUSelection::get_device_map();
  for(int i=0;i<s.num_devices;i++){
    s.gpu_data[i] = std::make_unique<gpu_profile_t>(devices[i]);
  }
  s.worker_trace.clear();
}

// Get elapsed time since initialization (microseconds)
long long profiler_mgpu::get_tick_since_start(){
  using namespace std::chrono;
  return duration_cast<microseconds>(steady_clock::now() - _starting_epoch).count();
}

// Get singleton multi-GPU profiler instance
profiler_mgpu& profiler_mgpu::Instance(){
  assert(single);
  return *single;
}

profiler_mgpu::~profiler_mgpu(){

}
} // namespace CESpGEMM
