#include"computeGPUAsync.h"
#include"decomp.cuh"
#include"computeGPUKernel.cuh"
#include<cub/device/device_scan.cuh>
#include"helper.cuh"
#include"profiler.h"
#include<exception>
#include<nvml.h>

// GPU memory management utilities
namespace GpuPtrManage{
// Allocate GPU memory
template<typename T>
static T* alloc(size_t n){
  T* ret;
  CHK_CUDA_ERR(cudaMalloc((void**)&ret, n*sizeof(T)));
  return ret;
}

// Free GPU memory
template<typename T>
static void dealloc(T* p){
  if(p) CHK_CUDA_ERR(cudaFree(p));
}

}


namespace CESpGEMM
{
// Check if buffer state is invalid (conflicting tags)
bool is_invalid(uint8_t s){
  if(s&TAG_KNL){
    if(s&TAG_H2D) return true;
    if(s&TAG_PTR) return true;
    if(s&TAG_DAT) return true;
  }
  return false;
}

// Modify pointer by setting LSB to encode buffer ID
void* modified_address(void*ptr, bool val){
  uint64_t pval = (uint64_t)ptr;
  CHK_ASSERT(pval % 8 == 0);
  return (void*)(pval | val);
}

// Multi-GPU callback factory for async operations

template<class GpuComputerT, class GControlT>
struct CallbackFactoryMGPU
{
// Callback after host-to-device transfer completes
static void finish_h2d(void* this_ptr){
  uint64_t ptrval = (uint64_t)this_ptr;
  bool bid = ptrval & 1;
  GpuComputerT *This = static_cast<GpuComputerT*> ((void*)(ptrval ^ bid));
  This->prof->timer_h2d.avail = true;
  GControlT &gcb = This->gcb;
  CHK_ASSERT(!gcb.knl_info.pending);
  gcb.knl_info.pending = true;
  gcb.buffer_status[bid] &= ~TAG_H2D;
  gcb.free_state |= TAG_H2D;
}


// Callback after GPU kernel completes
static void finish_kernel(void* this_ptr){
  uint64_t ptrval = (uint64_t)this_ptr;
  bool bid = ptrval & 1;
  GpuComputerT *This = static_cast<GpuComputerT*> ((void*)(ptrval ^ bid));
  GControlT &gcb = This->gcb;
  This->prof->timer_knl.avail = true;
  CHK_ASSERT(!gcb.ptr_info.pending);
  gcb.ptr_info.pending = true;
  gcb.buffer_status[bid] &= ~TAG_KNL;
  gcb.free_state |= TAG_KNL;
}

// Callback after pointer transfer from device to host completes
static void finish_ptr_d2h(void* this_ptr){
  uint64_t ptrval = (uint64_t)this_ptr;
  bool bid = ptrval & 1;
  GpuComputerT *This = static_cast<GpuComputerT*> ((void*)(ptrval ^ bid));
  GControlT &gcb = This->gcb;
  This->prof->timer_d2h.avail = true;
  CHK_ASSERT(!gcb.data_info.pending);
  gcb.data_info.pending = true;
  gcb.buffer_status[bid] &= ~TAG_PTR;
  gcb.free_state |= TAG_PTR;
}

// Callback after data transfer from device to host completes
static void finish_data_d2h(void* this_ptr){
  uint64_t ptrval = (uint64_t)this_ptr;
  bool bid = ptrval & 1;
  GpuComputerT *This = static_cast<GpuComputerT*> ((void*)(ptrval ^ bid));
  This->prof->timer_d2h.avail = true;
  uint64_t fin_info = This->gcb.finish_info;
  int cBlockId = fin_info >> 32;
  bool lbuffer = fin_info & 1; 
  bool gBuffer = (fin_info >> 1) & 1;
  CHK_ASSERT(bid == lbuffer); // assert the fin_info is not modified
  This->wt[gBuffer]->setBlockStatus(cBlockId, 64|This->device_id);
  This->gcb.buffer_status[lbuffer] &= ~TAG_DAT;
  This->gcb.free_state |= TAG_DAT;
  CHK_ASSERT(This->finish_counter!=nullptr);
  This->finish_counter->fetch_add(1, std::memory_order::memory_order_relaxed);
}

} ;


// Static member: temporary buffer size for CUB operations
template<class Gs_t, bool Single>
size_t GpuComputerAsync<Gs_t, Single>::cub_tmp_bytes = 4096;

// Static member: global memory limit for GPU allocation
template<class Gs_t, bool Single>
size_t GpuComputerAsync<Gs_t, Single>::global_memory_limit = ULLONG_MAX;

template<class Gs_t, bool Single>
void GpuComputerAsync<Gs_t, Single>::set_call_back_finish_counter(std::atomic_llong*counter){
  if(this->finish_counter != nullptr){
    printf("%p\n", this->finish_counter);
  }
  CHK_ASSERT(this->finish_counter == nullptr);
  // CHK_ASSERT_EQL(this->finish_counter, nullptr);
  this->finish_counter = counter;
}

template<class Gs_t, bool Single>
// Check if GPU has enough memory for required allocations
bool GpuComputerAsync<Gs_t, Single>::check_device_memory_available(int p_nrA, int p_ncB, size_t p_max_nnzA, size_t p_max_nnzB, unsigned long long p_max_flop, const Gs_t* gs_ptr){
  cudaSetDevice(this->device_id); // Ensure we are checking the correct device

  size_t total_required_bytes = 0;

  // Calculate memory requirements for compressed data structures
  int seglen = gs_ptr->cmpInfo.seglen;
  int nrB_dim = gs_ptr->csrB_T->nc; // Number of rows in B is num cols in B_T

  size_t anchorLen_bytes = static_cast<size_t>(((nrB_dim + 1) / seglen + 10) * 3 * sizeof(uint32_t));
  size_t dataLen_bytes = static_cast<size_t>(nrB_dim * 2 + 1 + 512); // Assuming uint8_t, so count is bytes
  size_t controlLen_bytes = static_cast<size_t>(nrB_dim / 8 + 1 + 64); // Assuming uint8_t

  // Memory for double-buffered GPU data
  for (int i = 0; i < 2; ++i) {
      total_required_bytes += (p_nrA + 1) * sizeof(SrcEType); // ptrA
      total_required_bytes += p_max_nnzA * sizeof(IdxType);   // idxA
      total_required_bytes += p_max_nnzA * sizeof(ValType);   // valA
      total_required_bytes += (nrB_dim + 1) * sizeof(EIdType); // ptrB
      total_required_bytes += p_max_nnzB * sizeof(IdxType);   // idxB
      total_required_bytes += p_max_nnzB * sizeof(ValType);   // valB
      total_required_bytes += anchorLen_bytes;                 // anchor
      total_required_bytes += dataLen_bytes;                   // data
      total_required_bytes += controlLen_bytes;                // control
      total_required_bytes += p_max_flop * sizeof(IdxType);   // tmpIdx1
      total_required_bytes += p_max_flop * sizeof(ValType);   // tmpVal1
      total_required_bytes += (p_nrA + 1) * sizeof(EIdType);  // ptrC
  }

  // Memory for accumulator and temporary buffers
  total_required_bytes += static_cast<size_t>(nGridsMerge) * p_ncB * sizeof(ValType); // d_accumulator
  total_required_bytes += this->cub_tmp_bytes; // d_cubTempBuffer (already in bytes)
  total_required_bytes += (p_nrA + 1) * sizeof(EIdType);  // d_flop_count
  total_required_bytes += (p_nrA + 1) * sizeof(EIdType);  // d_flop_offset
  total_required_bytes += p_max_flop * sizeof(IdxType);   // d_tmpIdx2
  total_required_bytes += p_max_flop * sizeof(ValType);   // d_tmpVal2
  total_required_bytes += (p_nrA + 1) * sizeof(EIdType);  // d_nnzC

  // Query available GPU memory
  size_t free_mem = 0;
  size_t total_mem = 0;
  cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
  if(global_memory_limit != ULLONG_MAX) free_mem = min(free_mem, global_memory_limit);

  if (err != cudaSuccess) {
      std::cerr << "CUDA Error: cudaMemGetInfo failed: " << cudaGetErrorString(err) << std::endl;
      return false; // Cannot determine available memory
  }

  // std::cout << "Device ID: " << this->device_id << std::endl;
  // std::cout << "Available GPU Memory: " << free_mem / (1024.0 * 1024.0) << " MB" << std::endl;
  // std::cout << "Total GPU Memory: " << total_mem / (1024.0 * 1024.0) << " MB" << std::endl;
  // std::cout << "Required GPU Memory for GpuComputerAsync: " << total_required_bytes / (1024.0 * 1024.0) << " MB" << std::endl;

  // std::cout<< "flops = "<<p_max_flop*(sizeof(IdxType)+sizeof(ValType))*3ll<<"\n";
  // std::cout << "d_accumulator: "<<nGridsMerge*1ll*p_ncB*sizeof(ValType)/(1024.0*1024.0)<<"\n";
  std::cout << "GPU memory alloc: "<<total_required_bytes/(1024.0*1024.0)<<" / "<<free_mem/(1024.0*1024.0)<<"\n";

  if (free_mem < total_required_bytes) {
      std::cerr << "Error: Insufficient GPU memory available." << std::endl;
      std::cerr << "Required: " << total_required_bytes << " bytes, Available: " << free_mem << " bytes." << std::endl;
      // Optional: Detailed breakdown of largest allocations if debugging
      // size_t mgpu_buffer_one_set = 0;
      // mgpu_buffer_one_set += (p_nrA + 1) * sizeof(SrcEType); 
      // mgpu_buffer_one_set += p_max_nnzA * sizeof(IdxType);   
      // // ... and so on for one set of mgpu_buffer
      // std::cerr << "Per mgpu_buffer set: " << mgpu_buffer_one_set / (1024.0*1024.0) << " MB" << std::endl;
      // std::cerr << "d_accumulator: " << (static_cast<size_t>(nGridsMerge) * p_ncB * sizeof(ValType)) / (1024.0*1024.0) << " MB" << std::endl;
      // // ... etc.
      return false;
  }
  // std::cout << "Sufficient GPU memory available." << std::endl;
  return true;
}

template<class Gs_t, bool Single>
// Allocate GPU memory if enough space is available
bool GpuComputerAsync<Gs_t, Single>::try_initialize(size_t max_nnzA, size_t max_nnzB, size_t max_flop){
  if(check_device_memory_available(nrA, ncB, max_nnzA, max_nnzB, max_flop, gs) == false){
    return false;
  }
  int nrB = gs->csrB_T.get()->nc;
  initialized = true;
  try{
    // Allocate double-buffered GPU memory
    for(int i=0;i<2;i++){
      GpuBuffer<SrcEType, EIdType, ValType> & G = this->mgpu_buffer[i];
      G.ptrA = GpuPtrManage::alloc<SrcEType>(nrA+1);
      G.idxA = GpuPtrManage::alloc<IdxType>(max_nnzA);
      G.valA = GpuPtrManage::alloc<ValType>(max_nnzA);
      G.ptrB = GpuPtrManage::alloc<EIdType>(nrB+1);
      G.idxB = GpuPtrManage::alloc<IdxType>(max_nnzB);
      G.valB = GpuPtrManage::alloc<ValType>(max_nnzB);
      G.anchor = GpuPtrManage::alloc<uint32_t>(anchorLen) ;
      G.data = GpuPtrManage::alloc<uint8_t>(dataLen);
      G.control = GpuPtrManage::alloc<uint8_t>(controlLen) ;
      G.tmpIdx1 = GpuPtrManage::alloc<IdxType>(max_flop);
      G.tmpVal1 = GpuPtrManage::alloc<ValType>(max_flop);
      G.ptrC = GpuPtrManage::alloc<EIdType>(nrA+1);
    }
    // Allocate accumulator for merging results from multiple grids
    this->d_accumulator = GpuPtrManage::alloc<ValType>( nGridsMerge * ncB);
    CHK_CUDA_ERR( cudaMemset(d_accumulator, 0, sizeof(ValType)* nGridsMerge * ncB) );
    this->d_cubTempBuffer = GpuPtrManage::alloc<uint8_t>( cub_tmp_bytes );
    this->d_flop_count = GpuPtrManage::alloc<EIdType>(nrA+1); 
    this->d_flop_offset = GpuPtrManage::alloc<EIdType>(nrA+1);
    this->d_tmpIdx2 = GpuPtrManage::alloc<IdxType>(max_flop);
    this->d_tmpVal2 = GpuPtrManage::alloc<ValType>(max_flop);
    this->d_nnzC = GpuPtrManage::alloc<EIdType>(nrA + 1); 
  }
  catch(...){
    std::cerr<<"Exception during GPU resource allocation\n";
    return false;
  }
  this->max_allowed_flop = max_flop;
  return true;
}

// Bind current thread to CPU cores closest to specified GPU
static void bindThreadToGPUCPUAffinity(int device_id){
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(device_id, &device);
    unsigned long cpuSet[8] = {0};          // 最多 512 CPU
    int cpu_size = 8;                        // 每个 unsigned long 是 64 bit

    auto err = nvmlDeviceGetCpuAffinity(device, cpu_size, cpuSet);
    if(err == nvmlReturn_t::NVML_ERROR_NOT_SUPPORTED) {
        std::cerr << "WARNING! ===: nvmlDeviceGetCpuAffinity not supported\n ===";
        return;
    }
    CHK_ASSERT_EQL(err, nvmlReturn_t::NVML_SUCCESS);

    cpu_set_t cpuset; 
    CPU_ZERO(&cpuset);

    for (int i = 0; i < cpu_size; i++) {
        unsigned long mask = cpuSet[i];
        for (int b = 0; b < 64; b++) {
            if (mask & (1UL << b)) {
                int cpu_id = i * 64 + b;
                CPU_SET(cpu_id, &cpuset);
            }
        }
    }

    int ret = pthread_setaffinity_np(pthread_self(),
                                    sizeof(cpu_set_t),
                                    &cpuset);
    CHK_ASSERT_EQL(ret, 0);
}

template<class Gs_t, bool Single>
// Constructor: Initialize GPU computer with device and dimensions
GpuComputerAsync<Gs_t, Single>::GpuComputerAsync(int nrA, int ncB, size_t max_nnzA, size_t max_nnzB, int device, Gs_t *gs, std::vector<std::unique_ptr<write_t>> & write_data, gpu_profile_t* gpu_profiler) :gs(gs), nrA(nrA), ncB(ncB), device_id(device), counter(0), initialized(false){
  this->finish_counter = nullptr;
  cudaSetDevice(device);
  CHK_ASSERT_EQL(nvmlInit(), nvmlReturn_t::NVML_SUCCESS);
  bindThreadToGPUCPUAffinity(device);
  // if(check_device_memory_available(nrA, ncB, max_nnzA, max_nnzB, max_flop, gs) == 0){
  //   throw std::exception();
  //   return;
  // }
  this->prof = gpu_profiler;
  
  // Initialize all GPU pointers to nullptr
  for (int i = 0; i < 2; ++i) {
    auto& G = this->mgpu_buffer[i];
    G.ptrA = nullptr;
    G.idxA = nullptr;
    G.valA = nullptr;
    G.ptrB = nullptr;
    G.idxB = nullptr;
    G.valB = nullptr;
    G.anchor = nullptr;
    G.data = nullptr;
    G.control = nullptr;
    G.tmpIdx1 = nullptr;
    G.tmpVal1 = nullptr;
    G.ptrC = nullptr;
  }

  this->d_accumulator = nullptr;
  this->d_cubTempBuffer = nullptr;
  this->d_flop_count = nullptr;
  this->d_flop_offset = nullptr;
  this->d_tmpIdx2 = nullptr;
  this->d_tmpVal2 = nullptr;
  this->d_nnzC = nullptr;

  // Track which buffers have been loaded to GPU
  buffer_visited_h2d[0]=buffer_visited_h2d[1]=0;
  buffer_visited_decomp[0]=buffer_visited_decomp[1]=0;

  // Create CUDA streams for async operations
  cub_tmp_bytes = 4096;
  for(int i=0;i<3;i++) cudaStreamCreate(streams + i);
  this->gcb.buffer_status[0] = 0;
  this->gcb.buffer_status[1] = 0;
  this->gcb.knl_info.pending = false;
  this->gcb.ptr_info.pending = false;
  this->gcb.data_info.pending = false;
  this->gcb.free_state = 0b1111;
  

  // Calculate sizes for compressed data structures
  int seglen = this->gs->cmpInfo.seglen;
  int nrB = gs->csrB_T.get()->nc;
  
  anchorLen = ((nrB + 1)/seglen + 10) * 3 * sizeof(uint32_t);
  dataLen = nrB*2 + 1 + 512;
  controlLen = nrB /8+1+64;

  for(int b=0;b<2;b++){
    wt[b] = write_data[b].get();
  }
}

template<class Gs_t, bool Single>
// Destructor: Free all GPU resources
GpuComputerAsync<Gs_t, Single>::~GpuComputerAsync() {
  // Destroy CUDA streams
  for (int i = 0; i < 3; ++i) {
    if (streams[i]) {
      cudaStreamDestroy(streams[i]);
      streams[i] = nullptr;
    }
  }
    
  if(!initialized) return;

  // Free GPU buffers
  for (int i = 0; i < 2; ++i) {
    auto& G = this->mgpu_buffer[i];
    GpuPtrManage::dealloc(G.ptrA);      G.ptrA = nullptr;
    GpuPtrManage::dealloc(G.idxA);      G.idxA = nullptr;
    GpuPtrManage::dealloc(G.valA);      G.valA = nullptr;
    GpuPtrManage::dealloc(G.ptrB);      G.ptrB = nullptr;
    GpuPtrManage::dealloc(G.idxB);      G.idxB = nullptr;
    GpuPtrManage::dealloc(G.valB);      G.valB = nullptr;
    GpuPtrManage::dealloc(G.anchor);    G.anchor = nullptr;
    GpuPtrManage::dealloc(G.data);      G.data = nullptr;
    GpuPtrManage::dealloc(G.control);   G.control = nullptr;
    GpuPtrManage::dealloc(G.tmpIdx1);   G.tmpIdx1 = nullptr;
    GpuPtrManage::dealloc(G.tmpVal1);   G.tmpVal1 = nullptr;
    GpuPtrManage::dealloc(G.ptrC);      G.ptrC = nullptr;
  }

  // Free other GPU allocations
  GpuPtrManage::dealloc(this->d_accumulator);    this->d_accumulator = nullptr;
  GpuPtrManage::dealloc(this->d_cubTempBuffer);  this->d_cubTempBuffer = nullptr;
  GpuPtrManage::dealloc(this->d_flop_count);     this->d_flop_count = nullptr;
  GpuPtrManage::dealloc(this->d_flop_offset);    this->d_flop_offset = nullptr;
  GpuPtrManage::dealloc(this->d_tmpIdx2);        this->d_tmpIdx2 = nullptr;
  GpuPtrManage::dealloc(this->d_tmpVal2);        this->d_tmpVal2 = nullptr;
  GpuPtrManage::dealloc(this->d_nnzC);           this->d_nnzC = nullptr;
}

#define RET_ON_CUDA_ERR(val)        \
  do{                               \
    cudaError_t err__ = (val);      \
    if(err__ != cudaSuccess){       \
      return err__;                  \
    }                               \
  } while(0)                        \


template<class Gs_t, bool Single>
// Add a task to GPU: transfer data (H2D) asynchronously
cudaError_t GpuComputerAsync<Gs_t, Single>::addGpuTask(int rBid, int cBid, bool gBuffer){
  cudaSetDevice(this->device_id);
  // using callbacks = CudaCallBackFactory<EIdType>;
  using callbacks = CallbackFactoryMGPU<std::remove_pointer_t<decltype(this)>, decltype(this->gcb)>;
  bool local_buffer_id = (counter ++) &1;
  gcb.free_state &= ~TAG_H2D;
  CHK_ASSERT( !is_invalid( gcb.buffer_status[local_buffer_id] |= TAG_H2D ) );

  const csr<SrcEType, ValType> & csrA = *gs->csrA.get();
  const csc<SrcEType, ValType> &cscB = *gs->csrB_T.get();
  const raw_csr<ValType> &raw_csrB = *gs->vcsrb_raw[cBid].get();

  const compress_t & hcmp = gs->v_comp_ptr.at(cBid);
  IdxType rStart = rBid * nrA, rEnd = min(rStart + nrA, csrA.nr), block_nrA = rEnd - rStart;
  IdxType cStart = cBid * ncB, cEnd = min(cStart + ncB, cscB.nr), block_ncB = cEnd - cStart;
  EIdType nnzB = cscB.ptr[cEnd] - cscB.ptr[cStart];

  EIdType offset_start = csrA.ptr[rStart], offset_end = csrA.ptr[rEnd], nnzA = offset_end-offset_start;
  GpuBuffer<SrcEType, EIdType, ValType>& buffer = this->mgpu_buffer[local_buffer_id];
  prof->timer_h2d.record_start(streams[0]);

  // Only load B once if there's only one block
  if(gs->numBlocksB > 1 || !this->buffer_visited_h2d[local_buffer_id]){
    buffer_visited_h2d[local_buffer_id] = true;
    RET_ON_CUDA_ERR(cudaMemcpyAsync(buffer.idxB, raw_csrB.idx, sizeof(IdxType) * nnzB, cudaMemcpyHostToDevice, streams[0]));
    RET_ON_CUDA_ERR(cudaMemcpyAsync(buffer.valB, raw_csrB.val, sizeof(ValType) * nnzB, cudaMemcpyHostToDevice, streams[0]));
    CHK_ASSERT_LESS(sizeof(uint32_t) * 3 * (hcmp.num_segs + 1), (size_t)this->anchorLen);
    CHK_ASSERT_LESS(hcmp.bytes_of_data, this->dataLen);
    CHK_ASSERT_LESS(hcmp.nbytes_control, this->controlLen);
    RET_ON_CUDA_ERR(cudaMemcpyAsync(buffer.anchor, hcmp.anchor_data, sizeof(uint32_t) * 3 * (hcmp.num_segs + 1), cudaMemcpyHostToDevice, streams[0]));
    RET_ON_CUDA_ERR(cudaMemcpyAsync(buffer.data, hcmp.data, sizeof(uint8_t) * hcmp.bytes_of_data, cudaMemcpyHostToDevice, streams[0]));
    RET_ON_CUDA_ERR(cudaMemcpyAsync(buffer.control, hcmp.control, sizeof(uint8_t) * hcmp.nbytes_control, cudaMemcpyHostToDevice, streams[0]));
  }
  
  RET_ON_CUDA_ERR(cudaMemcpyAsync(buffer.ptrA, csrA.ptr + rStart, sizeof(EIdType) * (block_nrA+1), cudaMemcpyHostToDevice, streams[0]));
  RET_ON_CUDA_ERR(cudaMemcpyAsync(buffer.idxA, csrA.idx + offset_start, sizeof(IdxType) * (nnzA), cudaMemcpyHostToDevice, streams[0]));
  RET_ON_CUDA_ERR(cudaMemcpyAsync(buffer.valA, csrA.val + offset_start, sizeof(ValType) * (nnzA), cudaMemcpyHostToDevice, streams[0]));

  prof->kbytes_h2d += ((sizeof(IdxType)+sizeof(ValType)) * (nnzB+nnzA) + sizeof(uint32_t) * 3 * (hcmp.num_segs + 1) + hcmp.bytes_of_data + hcmp.nbytes_control + sizeof(EIdType)*(block_nrA + 1)) / 1024.0;

  prof->timer_h2d.record_end(streams[0]);
  cudaStreamWaitEvent(streams[0], prof->timer_h2d.end_point);

  CHK_ASSERT(!gcb.knl_info.pending);
  this->gcb.knl_info.setData(
    local_buffer_id, rBid, cBid, gBuffer, 
    block_nrA, block_ncB, offset_start, streams[1]
  ) ;
  // RET_ON_CUDA_ERR( cudaLaunchHostFunc(streams[0], callbacks::finish_h2d, modified_address(&this->gcb, local_buffer_id)) );
  RET_ON_CUDA_ERR( cudaLaunchHostFunc(streams[0], callbacks::finish_h2d, modified_address(this, local_buffer_id)) );
  return cudaSuccess;
}

template<class Gs_t, bool Single>
// Execute GPU kernel: decompress, count flops, expand, merge
cudaError_t GpuComputerAsync<Gs_t, Single>::doKernel(){
  cudaSetDevice(this->device_id);

  const int seg_len = this->gs->cmpInfo.seglen;
  const int nz_head = this->gs->cmpInfo.max_zero_seg;
  // using callbacks = CudaCallBackFactory<EIdType>;
  using callbacks = CallbackFactoryMGPU<std::remove_pointer_t<decltype(this)>, decltype(this->gcb)>;


  this->gcb.knl_info.pending = false;
  this->gcb.free_state &= ~TAG_KNL;
  const Kernel_MetaData<EIdType> &kmd = this->gcb.knl_info;
  CHK_ASSERT( !is_invalid( this->gcb.buffer_status[kmd.local_buffer_id] |= TAG_KNL ) ); 
  const compress_t & hcmp = this->gs->v_comp_ptr.at(kmd.cBid);
  GpuBuffer<SrcEType, EIdType, ValType> & buffer = this->mgpu_buffer[kmd.local_buffer_id];
  int row_b = gs->csrB_T.get()->nc;
  int block_nr = kmd.block_nr, block_nc = kmd.block_nc;
  
  prof->timer_knl.record_start(kmd.stream);

  // Only decompress once if there's only one block
  if(gs->numBlocksB > 1 || !this->buffer_visited_decomp[kmd.local_buffer_id]){
    CHK_CUDA_ERR(cudaMemsetAsync(buffer.ptrB, -1, sizeof(EIdType)*(row_b), kmd.stream));
    buffer_visited_decomp[kmd.local_buffer_id] = true;
    decomp_launch(buffer.anchor, buffer.data, buffer.control, buffer.ptrB, seg_len, nz_head, hcmp.num_segs, kmd.stream);
  }

  
  // Count flops for each row of A with decompressed B
  count_flops_comp<SrcEType, EIdType><<<block_nr, 512, 0, kmd.stream>>>(
    kmd.start_offset,
    buffer.ptrA, buffer.idxA,
    buffer.ptrB, this->d_flop_count
  ) ;
  
  CHK_CUDA_ERR( cub::DeviceScan::ExclusiveSum(this->d_cubTempBuffer, cub_tmp_bytes, this->d_flop_count, this->d_flop_offset, block_nr+1, (cudaStream_t)(kmd.stream)) );

  // Expand sparse multiplication to dense intermediate format
  expand_dense_h2d_comp<SrcEType, EIdType, ValType><<<block_nr, 512, 0, kmd.stream>>>(
    kmd.start_offset, buffer.ptrA, buffer.ptrB,
    buffer.idxA, buffer.idxB,
    buffer.valA, buffer.valB,
    this->d_flop_offset,
    buffer.tmpIdx1, buffer.tmpVal1
  ) ;

  CHK_CUDA_ERR(cudaGetLastError());

  // Merge intermediate results into global accumulator
  merge_interm_result_glb_acum<EIdType, ValType><<< nGridsMerge , 512, 0, kmd.stream>>>(
    block_nr, block_nc, 0,
    this->d_flop_offset,
    buffer.tmpIdx1, buffer.tmpVal1,
    this->d_accumulator,
    this->d_tmpIdx2, this->d_tmpVal2,
    this->d_nnzC
  );

  CHK_CUDA_ERR(cub::DeviceScan::ExclusiveSum(this->d_cubTempBuffer, cub_tmp_bytes, this->d_nnzC, buffer.ptrC, block_nr+1, kmd.stream));

  // Collect final sparse result from accumulator
  collect<EIdType, ValType><<<block_nr, 512, 0, kmd.stream>>>(
    this->d_flop_offset, buffer.ptrC,
    this->d_tmpIdx2, this->d_tmpVal2,
    buffer.tmpIdx1, buffer.tmpVal1
  ) ;

  prof->timer_knl.record_end(kmd.stream);

  CHK_CUDA_ERR(cudaGetLastError());
  CHK_ASSERT(!gcb.ptr_info.pending);
  this->gcb.ptr_info.setData(kmd.local_buffer_id, kmd.rBid, kmd.cBid, kmd.gBuffer, kmd.block_nr, streams[2]);
  // CHK_CUDA_ERR(cudaLaunchHostFunc(kmd.stream, callbacks::finish_kernel, modified_address(&this->gcb, kmd.local_buffer_id)));
  CHK_CUDA_ERR(cudaLaunchHostFunc(kmd.stream, callbacks::finish_kernel, modified_address(this, kmd.local_buffer_id)));
  return cudaSuccess;
}

template<class Gs_t, bool Single>
// Transfer result pointer array from device to host
cudaError_t GpuComputerAsync<Gs_t, Single>::doPtrD2H(){
  cudaSetDevice(this->device_id);
  // using callbacks = CudaCallBackFactory<EIdType>;
  using callbacks = CallbackFactoryMGPU<std::remove_pointer_t<decltype(this)>, decltype(this->gcb)>;
  this->gcb.ptr_info.pending = false;
  this->gcb.free_state &= ~TAG_PTR;
  const MetaData &pmd = gcb.ptr_info;
  CHK_ASSERT( !is_invalid( this->gcb.buffer_status[pmd.local_buffer_id] |= TAG_PTR ));

  GpuBuffer<SrcEType, EIdType, ValType> & buffer = this->mgpu_buffer[pmd.local_buffer_id];
  write_t &w_tgt = *this->wt[pmd.gBuffer];
  std::vector<EIdType> & hPtrC = w_tgt.getPtr(pmd.cBid);
  prof->timer_d2h.record_start(pmd.stream);
  CHK_CUDA_ERR(cudaMemcpyAsync(hPtrC.data(), buffer.ptrC, sizeof(EIdType)*(pmd.block_nr+1), cudaMemcpyDeviceToHost, pmd.stream) );
  CHK_ASSERT(!gcb.data_info.pending);
  prof->timer_d2h.record_end(pmd.stream);
  prof->kbytes_d2h += sizeof(EIdType)*(pmd.block_nr+1)/1024.0;

  this->gcb.data_info.setData(pmd.local_buffer_id, pmd.rBid, pmd.cBid, pmd.gBuffer, pmd.block_nr, streams[2]);
  // CHK_CUDA_ERR(cudaLaunchHostFunc(pmd.stream, callbacks::finish_ptr_d2h, modified_address(&this->gcb, pmd.local_buffer_id)));
  CHK_CUDA_ERR(cudaLaunchHostFunc(pmd.stream, callbacks::finish_ptr_d2h, modified_address(this, pmd.local_buffer_id)));
  return cudaSuccess;
}

template<class Gs_t, bool Single>
// Transfer result data (idx, val) from device to host
cudaError_t GpuComputerAsync<Gs_t, Single>::doDataD2H(){
  cudaSetDevice(this->device_id); 
  // using ccbf = CudaCallBackFactory<EIdType>;
  using callbacks = CallbackFactoryMGPU<std::remove_pointer_t<decltype(this)>, decltype(this->gcb)>;

  this->gcb.data_info.pending = false;
  this->gcb.free_state &= ~TAG_DAT;
  const MetaData &dmd = gcb.data_info;
  CHK_ASSERT( !is_invalid( this->gcb.buffer_status[dmd.local_buffer_id] |= TAG_DAT));

  GpuBuffer<SrcEType, EIdType, ValType> & buffer = this->mgpu_buffer[dmd.local_buffer_id];
  write_t &w_tgt = *this->wt[dmd.gBuffer];
  EIdType nnz = w_tgt.getPtr(dmd.cBid).at(dmd.block_nr) ;
  raw_csr<ValType> & tgt_data = w_tgt.getRawCsr(dmd.cBid);
  tgt_data.refresh_with_size(nnz);
  prof->timer_d2h.record_start(dmd.stream);
  // printf("nnz=%lu\n", nnz);
  CHK_CUDA_ERR(cudaMemcpyAsync(tgt_data.idx, buffer.tmpIdx1, sizeof(IdxType)*nnz, cudaMemcpyDeviceToHost, dmd.stream));
  CHK_CUDA_ERR(cudaMemcpyAsync(tgt_data.val, buffer.tmpVal1, sizeof(ValType)*nnz, cudaMemcpyDeviceToHost, dmd.stream));
  prof->timer_d2h.record_end(dmd.stream);
  prof->kbytes_d2h += (sizeof(IdxType) +sizeof(ValType)) *nnz;
  this->gcb.finish_info = ((ull)dmd.cBid << 32ull) | (dmd.gBuffer<<1) | dmd.local_buffer_id;
  
  // CHK_CUDA_ERR(cudaLaunchHostFunc(dmd.stream, &ccbf::template finish_data_d2h<Gs_t, Single>, modified_address(this, dmd.local_buffer_id)));
  CHK_CUDA_ERR(cudaLaunchHostFunc(dmd.stream, callbacks::finish_data_d2h, modified_address(this, dmd.local_buffer_id)));
  return cudaSuccess;
}

} // namespace CESpGEMM
