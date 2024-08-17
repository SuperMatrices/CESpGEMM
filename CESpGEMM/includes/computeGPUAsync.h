#pragma once
#include"Storage.h"
#include<cstdint>
#include<cuda.h>
#include<cuda_runtime.h>
#include"task_alloc.h"
#include"compressor.h"
#include"WritableStorage.h"
#include<atomic>

struct MetaData
{
  std::atomic_bool pending;
  bool local_buffer_id, gBuffer;
  int rBid, cBid, block_nr;
  cudaStream_t stream;
  MetaData():stream(0){
    
  }
  void setData(int local_buffer_id, int rid, int cid, bool gBuffer, int block_nr, cudaStream_t stream){
    this->local_buffer_id = local_buffer_id;
    this->rBid = rid;
    this->cBid = cid;
    this->gBuffer = gBuffer;
    this->block_nr = block_nr;
    this->stream = stream;
  }
  MetaData(const MetaData&)=delete;
  // MetaData(bool pend, int local_buffer_id, int rid, int cid, bool gBuffer, int block_nr, cudaStream_t stream): pending(std::atomic_bool(pend)), local_buffer_id(local_buffer_id), rBid(rid), cBid(cid), gBuffer(gBuffer), block_nr(block_nr), stream(stream)
  // {

  // }
} ;

template<typename EIdType>
struct Kernel_MetaData : MetaData{
  int block_nc;
  EIdType start_offset;
  Kernel_MetaData(){

  }
  void setData(int local_buffer_id, int rid, int cid, bool gBuffer, int block_nr, int block_nc, EIdType start_offset, cudaStream_t stream){
    this->local_buffer_id = local_buffer_id;
    this->rBid = rid;
    this->cBid = cid;
    this->gBuffer = gBuffer;
    this->block_nr = block_nr;
    this->stream = stream;
    this->block_nc = block_nc;
    this->start_offset = start_offset;
  }
  // Kernel_MetaData(bool pend, int local_buffer_id, int rid, int cid, bool gBuffer, int block_nr, int block_nc, EIdType offset, cudaStream_t stream): MetaData(pend, local_buffer_id, rid, cid, gBuffer, block_nr, stream)
  // {
  //   this->block_nc = block_nc;
  //   this->start_offset = offset;
  // }
} ;




namespace CESpGEMM
{

constexpr uint8_t TAG_H2D = 0b1000;
constexpr uint8_t TAG_KNL = 0b0100;
constexpr uint8_t TAG_PTR = 0b0010;
constexpr uint8_t TAG_DAT = 0b0001;
constexpr uint8_t TAG_ALL = 0b1111;


template<typename EIdType>
struct alignas(8) GPUControllingBlock
{
  std::atomic_uchar free_state;
  std::atomic_uchar buffer_status[2];
  // uint8_t free_state;
  // uint8_t buffer_status[2];
  Kernel_MetaData<EIdType> knl_info;
  MetaData ptr_info;
  MetaData data_info;
  GPUControllingBlock(const GPUControllingBlock&)=delete;
  GPUControllingBlock(){
    
  }
} ;


template<typename SrcEType,typename EIdType, typename ValType>
struct GpuBuffer{
  SrcEType*ptrA/*in/ker*/;
  EIdType *ptrB/*in/ker*/;
  IdxType*idxA/*in/ker*/, *idxB/*in/ker*/;
  ValType*valA/*in/ker*/, *valB/*in/ker*/;
  uint32_t*anchor/*in/ker*/;  
  uint8_t*data/*in/ker*/, *control/*in/ker*/;
  IdxType*tmpIdx1/*ker/out*/;
  ValType*tmpVal1/*ker/out*/;
  EIdType*ptrC/*ker/out*/;

  void realloc_size(size_t flops);
} ;




template<class Gs_t, bool SingleBlock>
class GpuComputerAsync
{
  using EIdType = typename Gs_t::eidType;
  using ValType = typename Gs_t::valType;
  using SrcEType = typename Gs_t::srcEtype;

  using write_t = ComputeResultData<Gs_t, SingleBlock>;
  using task_fetch_t = TaskFetch<2, int>;

public:
  static constexpr int nGridsMerge = 256;
  static size_t cub_tmp_bytes;

  void addGpuTask(int rBid, int cBid, bool gBuffer);
  void doKernel();
  void doPtrD2H();
  void doDataD2H();
  // void realloc_size(size_t flop);
  ~GpuComputerAsync();
  GpuComputerAsync(int nrA, int ncB, size_t max_nnzA, size_t max_nnzB, int device,
    Gs_t*gs, ull max_flop, std::vector<std::unique_ptr<write_t>> & write_data);
  GPUControllingBlock<EIdType> gcb;
  int counter;
  write_t* wt[2];
  int device_id;

private:
  int nrA, ncB;
  ull current_max_flop;
  Gs_t *gs;
  cudaStream_t streams[3];
  IdxType dataLen, controlLen, anchorLen;
  GpuBuffer<SrcEType, EIdType, ValType> mgpu_buffer[2];
  ValType* d_accumulator;
  void* d_cubTempBuffer;
  EIdType* d_flop_offset, *d_flop_count, *d_nnzC;
  IdxType* d_tmpIdx2;
  ValType* d_tmpVal2;


} ;


template class GpuComputerAsync<default_gs_t, false>;
template class GpuComputerAsync<default_gs_t, true>;

template class GpuComputerAsync<large_gs_t, false>;
template class GpuComputerAsync<large_gs_t, true>;


}//namespace CESpGEMM