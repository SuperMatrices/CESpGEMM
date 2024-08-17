#include"computeGPUAsync.h"
#include"decomp.cuh"
#include"computeGPUKernel.cuh"
#include<cub/device/device_scan.cuh>
#include"helper.cuh"
#include"profiler.h"

namespace GpuPtrManage{
template<typename T>
static T* alloc(size_t n){
  T* ret;
  CHK_CUDA_ERR(cudaMalloc((void**)&ret, n*sizeof(T)));
  return ret;
}

template<typename T>
static void dealloc(T* p){
  if(p) CHK_CUDA_ERR(cudaFree(p));
}

}




namespace CESpGEMM
{
template<typename SrcEType,typename EIdType, typename ValType>
void GpuBuffer<SrcEType, EIdType, ValType>::realloc_size(size_t flops){
  CHK_CUDA_ERR(cudaFree(tmpIdx1));
  CHK_CUDA_ERR(cudaFree(tmpVal1));
  CHK_CUDA_ERR(cudaMalloc(&tmpIdx1, sizeof(IdxType)*flops));
  CHK_CUDA_ERR(cudaMalloc(&tmpVal1, sizeof(ValType)*flops));
}

bool is_invalid(uint8_t s){
  if(s&TAG_KNL){
    if(s&TAG_H2D) return true;
    if(s&TAG_PTR) return true;
    if(s&TAG_DAT) return true;
  }
  return false;
}

void* modified_address(void*ptr, bool val){
  uint64_t pval = (uint64_t)ptr;
  CHK_ASSERT(pval % 8 == 0);
  return (void*)(pval | val);
}

template<typename EIdType>
struct CudaCallBackFactory{

static void finish_h2d(void* ptr){
  // CHK_CUDA_ERR(err);
  profiler::Instance().timer_h2d.avail=true;
  uint64_t ptrval = (uint64_t)ptr;
  bool bid = ptrval & 1;
  GPUControllingBlock<EIdType>* gcb = static_cast<GPUControllingBlock<EIdType>*> ((void*)(ptrval ^ bid));
  CHK_ASSERT(!gcb->knl_info.pending);
  gcb->knl_info.pending = true;
  gcb->buffer_status[bid] &= ~TAG_H2D;
  gcb->free_state |= TAG_H2D;
}

static void finish_kernel(void* ptr){
  profiler::Instance().timer_knl.avail=true;

// static void finish_kernel(cudaStream_t stream, cudaError_t err, void* ptr){
  // CHK_CUDA_ERR(err);
  //printf("###finish kernel\n");
  uint64_t ptrval = (uint64_t)ptr;
  bool bid = ptrval & 1;
  GPUControllingBlock<EIdType>* gcb = static_cast<GPUControllingBlock<EIdType>*> ((void*)(ptrval ^ bid));
  CHK_ASSERT(!gcb->ptr_info.pending);
  gcb->ptr_info.pending = true;
  gcb->buffer_status[bid] &= ~TAG_KNL;
  gcb->free_state |= TAG_KNL;
  //printf("####finish knl freestate=%d\n", gcb->free_state.load());
}


static void finish_ptr_d2h(void* ptr){
  profiler::Instance().timer_d2h.avail=true;

// static void finish_ptr_d2h(cudaStream_t stream, cudaError_t err, void* ptr){
  // CHK_CUDA_ERR(err);
  // //printf("###finish ptrd2h\n");
  uint64_t ptrval = (uint64_t)ptr;
  bool bid = ptrval & 1;
  GPUControllingBlock<EIdType>* gcb = static_cast<GPUControllingBlock<EIdType>*> ((void*)(ptrval ^ bid));
  CHK_ASSERT(!gcb->data_info.pending);
  gcb->data_info.pending = true;
  gcb->buffer_status[bid] &= ~TAG_PTR;
  //printf("###finish ptrd2h: buf_sta[%d]=%d\n", bid, gcb->buffer_status[bid].load());
  gcb->free_state |= TAG_PTR;
  //printf("####finish ptrd2h freestate=%d\n", gcb->free_state.load());
}


} ;

struct FinishData{
  void * ptr;
  int seg1;
  int seg2;
} ;

template<class Gs_t, bool Single>
static void finish_data_d2h(void* ptr){
  profiler::Instance().timer_d2h.avail=true;
// static void finish_data_d2h(cudaStream_t stream, cudaError_t err, void* ptr){
  // CHK_CUDA_ERR(err);
  //printf("###finish data_d2h\n");
  FinishData* cbd = (FinishData*)ptr;
  GpuComputerAsync<Gs_t, Single> * This = static_cast<GpuComputerAsync<Gs_t, Single> *>(cbd->ptr);
  int bufferId = cbd->seg1;
  bool lbuffer = bufferId & 1, gBuffer = bufferId >> 1;
  int cBlockId = cbd->seg2;
  ComputeResultData<Gs_t, Single> &res = * This->wt[gBuffer];
  res.setBlockStatus(cBlockId, 64|This->device_id);
  This->gcb.buffer_status[lbuffer] &= ~TAG_DAT;
  This->gcb.free_state |= TAG_DAT;
  //printf("####finish datad2h freestate=%d\n", This->gcb.free_state.load());
}


template<class Gs_t, bool Single>  
size_t GpuComputerAsync<Gs_t, Single>::cub_tmp_bytes = 4096;

template<class Gs_t, bool Single>
GpuComputerAsync<Gs_t, Single>::GpuComputerAsync(int nrA, int ncB, size_t max_nnzA, size_t max_nnzB, int device, Gs_t *gs, ull max_flop, std::vector<std::unique_ptr<write_t>> & write_data) :gs(gs), nrA(nrA), ncB(ncB), device_id(device), current_max_flop(max_flop), counter(0){
  cudaSetDevice(device);
  cub_tmp_bytes = 4096;
  for(int i=0;i<3;i++) cudaStreamCreate(streams + i);
  this->gcb.buffer_status[0] = 0;
  this->gcb.buffer_status[1] = 0;
  this->gcb.knl_info.pending = false;
  this->gcb.ptr_info.pending = false;
  this->gcb.data_info.pending = false;
  this->gcb.free_state = 0b1111;

  int nrB = gs->csrB_T.nc;
  
  anchorLen = ((nrB + 1)/compress_t::SegLen + 10) * 3 * sizeof(uint32_t);
  dataLen = nrB*2 + 1 + 512;
  controlLen = nrB /8+1+64;
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
  this->d_accumulator = GpuPtrManage::alloc<ValType>( nGridsMerge * ncB);
  cudaMemset(d_accumulator, 0, sizeof(ValType)* nGridsMerge * ncB) ;
  this->d_cubTempBuffer = GpuPtrManage::alloc<uint8_t>( cub_tmp_bytes );
  this->d_flop_count = GpuPtrManage::alloc<EIdType>(nrA+1); 
  this->d_flop_offset = GpuPtrManage::alloc<EIdType>(nrA+1);
  this->d_tmpIdx2 = GpuPtrManage::alloc<IdxType>(max_flop);
  this->d_tmpVal2 = GpuPtrManage::alloc<ValType>(max_flop);
  this->d_nnzC = GpuPtrManage::alloc<EIdType>(nrA + 1);
  for(int b=0;b<2;b++){
    wt[b] = write_data[b].get();
  }
}

template<class Gs_t, bool Single>
GpuComputerAsync<Gs_t, Single>::~GpuComputerAsync(){
  for(int i=0;i<3;i++) cudaStreamDestroy(streams[i]);
  for(int i=0;i<2;i++){
    GpuBuffer<SrcEType, EIdType, ValType> & G = this->mgpu_buffer[i];
    GpuPtrManage::dealloc(G.ptrA);
    GpuPtrManage::dealloc(G.idxA);
    GpuPtrManage::dealloc(G.valA);
    GpuPtrManage::dealloc(G.ptrB);
    GpuPtrManage::dealloc(G.idxB);
    GpuPtrManage::dealloc(G.valB);
    GpuPtrManage::dealloc(G.anchor);
    GpuPtrManage::dealloc(G.data);
    GpuPtrManage::dealloc(G.control);
    GpuPtrManage::dealloc(G.tmpIdx1);
    GpuPtrManage::dealloc(G.tmpVal1);
    GpuPtrManage::dealloc(G.ptrC);
  }
  GpuPtrManage::dealloc(this->d_accumulator);
  GpuPtrManage::dealloc(this->d_cubTempBuffer);
  GpuPtrManage::dealloc(this->d_flop_count);
  GpuPtrManage::dealloc(this->d_flop_offset);
  GpuPtrManage::dealloc(this->d_tmpIdx2);
  GpuPtrManage::dealloc(this->d_tmpVal2);
  GpuPtrManage::dealloc(this->d_nnzC);
}

// template<class Gs_t, bool Single>
// void GpuComputerAsync<Gs_t, Single>::realloc_size(size_t flop){
//   if(current_max_flop >= flop) return;
//   current_max_flop = flop;
//   for(int i=0;i<2;i++){
//     this->mgpu_buffer[i].realloc_size(flop);
//   }
//   GpuPtrManage::dealloc(this->d_tmpIdx2);
//   GpuPtrManage::dealloc(this->d_tmpVal2);
//   this->d_tmpIdx2 = GpuPtrManage::alloc<IdxType>(flop);
//   this->d_tmpVal2 = GpuPtrManage::alloc<ValType>(flop);
// }

template<class Gs_t, bool Single>
void GpuComputerAsync<Gs_t, Single>::addGpuTask(int rBid, int cBid, bool gBuffer){
  using ccbf = CudaCallBackFactory<EIdType>;
  bool local_buffer_id = (counter ++) &1;
  static bool buffer_visit[2]={0,0};
  gcb.free_state &= ~TAG_H2D;
  CHK_ASSERT( !is_invalid( gcb.buffer_status[local_buffer_id] |= TAG_H2D ) );
  //printf("###addGPUTask: (%d,%d,%d,%d)\n", rBid, cBid, gBuffer, local_buffer_id);

  csr<SrcEType, ValType> & csrA = gs->csrA;
  const csc<SrcEType, ValType> &cscB = gs->csrB_T;
  const raw_csr<ValType> &raw_csrB = *gs->vcsrb_raw[cBid].get();

  const compress_t & hcmp = gs->v_comp_ptr.at(cBid);
  IdxType rStart = rBid * nrA, rEnd = min(rStart + nrA, csrA.nr), block_nrA = rEnd - rStart;
  IdxType cStart = cBid * ncB, cEnd = min(cStart + ncB, cscB.nr), block_ncB = cEnd - cStart;
  EIdType nnzB = cscB.ptr[cEnd] - cscB.ptr[cStart];

  // printf("cbid=%d, has nnzB=%lld\n", cBid, nnzB);

  EIdType offset_start = csrA.ptr[rStart], offset_end = csrA.ptr[rEnd], nnzA = offset_end-offset_start;
  GpuBuffer<SrcEType, EIdType, ValType>& buffer = this->mgpu_buffer[local_buffer_id];
  profiler::Instance().timer_h2d.record_start(streams[0]);

  //if there is only one block of B, then each buffer just need to load once
  if(gs->numBlocksB > 1 || !buffer_visit[local_buffer_id]){
    buffer_visit[local_buffer_id] = true;
    CHK_CUDA_ERR(cudaMemcpyAsync(buffer.idxB, raw_csrB.idx, sizeof(IdxType) * nnzB, cudaMemcpyHostToDevice, streams[0]));
    CHK_CUDA_ERR(cudaMemcpyAsync(buffer.valB, raw_csrB.val, sizeof(ValType) * nnzB, cudaMemcpyHostToDevice, streams[0]));
    CHK_ASSERT_LESS(sizeof(uint32_t) * 3 * (hcmp.num_segs + 1), (size_t)this->anchorLen);
    CHK_ASSERT_LESS(hcmp.bytes_of_data, this->dataLen);
    CHK_ASSERT_LESS(hcmp.nbytes_control, this->controlLen);
    CHK_CUDA_ERR(cudaMemcpyAsync(buffer.anchor, hcmp.anchor_data, sizeof(uint32_t) * 3 * (hcmp.num_segs + 1), cudaMemcpyHostToDevice, streams[0]));
    CHK_CUDA_ERR(cudaMemcpyAsync(buffer.data, hcmp.data, sizeof(uint8_t) * hcmp.bytes_of_data, cudaMemcpyHostToDevice, streams[0]));
    CHK_CUDA_ERR(cudaMemcpyAsync(buffer.control, hcmp.control, sizeof(uint8_t) * hcmp.nbytes_control, cudaMemcpyHostToDevice, streams[0]));
  }
  
  CHK_CUDA_ERR(cudaMemcpyAsync(buffer.ptrA, csrA.ptr + rStart, sizeof(EIdType) * (block_nrA+1), cudaMemcpyHostToDevice, streams[0]));
  CHK_CUDA_ERR(cudaMemcpyAsync(buffer.idxA, csrA.idx + offset_start, sizeof(IdxType) * (nnzA), cudaMemcpyHostToDevice, streams[0]));
  CHK_CUDA_ERR(cudaMemcpyAsync(buffer.valA, csrA.val + offset_start, sizeof(ValType) * (nnzA), cudaMemcpyHostToDevice, streams[0]));

  profiler::Instance().kbytes_h2d += ((sizeof(IdxType)+sizeof(ValType)) * (nnzB+nnzA) + sizeof(uint32_t) * 3 * (hcmp.num_segs + 1) + hcmp.bytes_of_data + hcmp.nbytes_control + sizeof(EIdType)*(block_nrA + 1)) / 1024.0;

  profiler::Instance().timer_h2d.record_end(streams[0]);
  cudaStreamWaitEvent(streams[0], profiler::Instance().timer_h2d.end_point);

  CHK_ASSERT(!gcb.knl_info.pending);
  this->gcb.knl_info.setData(
    local_buffer_id, rBid, cBid, gBuffer, 
    block_nrA, block_ncB, offset_start, streams[1]
  ) ;
  CHK_CUDA_ERR( cudaLaunchHostFunc(streams[0], ccbf::finish_h2d, modified_address(&this->gcb, local_buffer_id)) );
}

template<class Gs_t, bool Single>
void GpuComputerAsync<Gs_t, Single>::doKernel(){
  static bool buffer_visit[2] = {0,0};
  using ccbf = CudaCallBackFactory<EIdType>;

  this->gcb.knl_info.pending = false;
  this->gcb.free_state &= ~TAG_KNL;
  const Kernel_MetaData<EIdType> &kmd = this->gcb.knl_info;
  CHK_ASSERT( !is_invalid( this->gcb.buffer_status[kmd.local_buffer_id] |= TAG_KNL ) ); 
  //printf("###doKernel: (%d,%d,%d,%d), gcb=%p, blocknr=%d\n", kmd.rBid, kmd.cBid, kmd.gBuffer, kmd.local_buffer_id, &this->gcb, kmd.block_nr);
  const compress_t & hcmp = this->gs->v_comp_ptr.at(kmd.cBid);
  GpuBuffer<SrcEType, EIdType, ValType> & buffer = this->mgpu_buffer[kmd.local_buffer_id];
  int row_b = gs->csrB_T.nc;
  int block_nr = kmd.block_nr, block_nc = kmd.block_nc;

  profiler::Instance().timer_knl.record_start(kmd.stream);

  //if there is only one block of B, then each buffer just need to load once
  if(gs->numBlocksB > 1 || !buffer_visit[kmd.local_buffer_id]){
    CHK_CUDA_ERR(cudaMemsetAsync(buffer.ptrB, -1, sizeof(EIdType)*(row_b), kmd.stream));
    buffer_visit[kmd.local_buffer_id] = true;
    decomp_knl<512, EIdType><<<hcmp.num_segs, 512, 0, kmd.stream>>>(
      buffer.anchor, buffer.data, buffer.control,
      buffer.ptrB
    );
  }
  
  count_flops_comp<SrcEType, EIdType><<<block_nr, 512, 0, kmd.stream>>>(
    kmd.start_offset, 
    buffer.ptrA, buffer.idxA,
    buffer.ptrB, this->d_flop_count
  ) ;

  CHK_CUDA_ERR( cub::DeviceScan::ExclusiveSum(this->d_cubTempBuffer, cub_tmp_bytes, this->d_flop_count, this->d_flop_offset, block_nr+1, (cudaStream_t)(kmd.stream)) );

  expand_dense_h2d_comp<SrcEType, EIdType, ValType><<<block_nr, 512, 0, kmd.stream>>>(
    kmd.start_offset, buffer.ptrA, buffer.ptrB,
    buffer.idxA, buffer.idxB,
    buffer.valA, buffer.valB,
    this->d_flop_offset,
    buffer.tmpIdx1, buffer.tmpVal1
  ) ;
  
  CHK_CUDA_ERR(cudaGetLastError());

  merge_interm_result_glb_acum<EIdType, ValType><<< nGridsMerge , 512, 0, kmd.stream>>>(
    block_nr, block_nc, 0,
    this->d_flop_offset,
    buffer.tmpIdx1, buffer.tmpVal1,
    this->d_accumulator,
    this->d_tmpIdx2, this->d_tmpVal2,
    this->d_nnzC
  );

  CHK_CUDA_ERR(cub::DeviceScan::ExclusiveSum(this->d_cubTempBuffer, cub_tmp_bytes, this->d_nnzC, buffer.ptrC, block_nr+1, kmd.stream));

  
  collect<EIdType, ValType><<<block_nr, 512, 0, kmd.stream>>>(
    this->d_flop_offset, buffer.ptrC, 
    this->d_tmpIdx2, this->d_tmpVal2,
    buffer.tmpIdx1, buffer.tmpVal1
  ) ; 
  profiler::Instance().timer_knl.record_end(kmd.stream);

  CHK_CUDA_ERR(cudaGetLastError());
  CHK_ASSERT(!gcb.ptr_info.pending);
  // this->gcb.ptr_info=MetaData(false, kmd.local_buffer_id, kmd.rBid, kmd.cBid, kmd.gBuffer, kmd.block_nr, streams[2]);
  this->gcb.ptr_info.setData(kmd.local_buffer_id, kmd.rBid, kmd.cBid, kmd.gBuffer, kmd.block_nr, streams[2]);
  // CHK_CUDA_ERR(cudaStreamAddCallback(kmd.stream, ccbf::finish_kernel, modified_address(&this->gcb, kmd.local_buffer_id), 0));
  CHK_CUDA_ERR(cudaLaunchHostFunc(kmd.stream, ccbf::finish_kernel, modified_address(&this->gcb, kmd.local_buffer_id)));
  
}

template<class Gs_t, bool Single>
void GpuComputerAsync<Gs_t, Single>::doPtrD2H(){
  using ccbf = CudaCallBackFactory<EIdType>;
  this->gcb.ptr_info.pending = false;
  this->gcb.free_state &= ~TAG_PTR;
  const MetaData &pmd = gcb.ptr_info;
  //printf("###doPtrD2H: (%d,%d,%d,%d)\n", pmd.rBid, pmd.cBid, pmd.gBuffer, pmd.local_buffer_id);
  CHK_ASSERT( !is_invalid( this->gcb.buffer_status[pmd.local_buffer_id] |= TAG_PTR ));

  GpuBuffer<SrcEType, EIdType, ValType> & buffer = this->mgpu_buffer[pmd.local_buffer_id];
  write_t &w_tgt = *this->wt[pmd.gBuffer];
  std::vector<EIdType> & hPtrC = w_tgt.getPtr(pmd.cBid);
  profiler::Instance().timer_d2h.record_start(pmd.stream);
  CHK_CUDA_ERR(cudaMemcpyAsync(hPtrC.data(), buffer.ptrC, sizeof(EIdType)*(pmd.block_nr+1), cudaMemcpyDeviceToHost, pmd.stream) );
  CHK_ASSERT(!gcb.data_info.pending);
  profiler::Instance().timer_d2h.record_end(pmd.stream);
  profiler::Instance().kbytes_d2h += sizeof(EIdType)*(pmd.block_nr+1)/1024.0;

  // this->gcb.data_info=MetaData(false, pmd.local_buffer_id, pmd.rBid, pmd.cBid, pmd.gBuffer, pmd.block_nr, streams[2]);
  this->gcb.data_info.setData(pmd.local_buffer_id, pmd.rBid, pmd.cBid, pmd.gBuffer, pmd.block_nr, streams[2]);
  // CHK_CUDA_ERR(cudaStreamAddCallback(pmd.stream, ccbf::finish_ptr_d2h, modified_address(&this->gcb, pmd.local_buffer_id), 0));
  CHK_CUDA_ERR(cudaLaunchHostFunc(pmd.stream, ccbf::finish_ptr_d2h, modified_address(&this->gcb, pmd.local_buffer_id)));
}

template<class Gs_t, bool Single>
void GpuComputerAsync<Gs_t, Single>::doDataD2H(){
  static FinishData call_back_data;
  using ccbf = CudaCallBackFactory<EIdType>;
  this->gcb.data_info.pending = false;
  this->gcb.free_state &= ~TAG_DAT;
  const MetaData &dmd = gcb.data_info;
  CHK_ASSERT( !is_invalid( this->gcb.buffer_status[dmd.local_buffer_id] |= TAG_DAT));
  //printf("###doDataD2H: (%d,%d,%d,%d)\n", dmd.rBid, dmd.cBid, dmd.gBuffer, dmd.local_buffer_id);

  GpuBuffer<SrcEType, EIdType, ValType> & buffer = this->mgpu_buffer[dmd.local_buffer_id];
  write_t &w_tgt = *this->wt[dmd.gBuffer];
  EIdType nnz = w_tgt.getPtr(dmd.cBid).at(dmd.block_nr) ;
  // printf("computeGPUAsync: (%d,%d):nnz=%lld\n", dmd.rBid, dmd.cBid, nnz);
  raw_csr<ValType> & tgt_data = w_tgt.getRawCsr(dmd.cBid);
  // refresh_buffer(tgt_data);
  //printf("###nnz=%d, refresh!\n", nnz);
  CHK_ASSERT_LESS((ull)nnz, gs->max_rcblock_flop);
  tgt_data.refresh_with_size(nnz);

  //printf("###cbid%d:, memcpyasync:%p,%p,%p,%p, %lld, %p\n", dmd.cBid, tgt_data.idx, tgt_data.val, buffer.tmpIdx1, buffer.tmpVal1, sizeof(IdxType)*nnz, dmd.stream);
  profiler::Instance().timer_d2h.record_start(dmd.stream);

  CHK_CUDA_ERR(cudaMemcpyAsync(tgt_data.idx, buffer.tmpIdx1, sizeof(IdxType)*nnz, cudaMemcpyDeviceToHost, dmd.stream));
  CHK_CUDA_ERR(cudaMemcpyAsync(tgt_data.val, buffer.tmpVal1, sizeof(ValType)*nnz, cudaMemcpyDeviceToHost, dmd.stream));
  profiler::Instance().timer_d2h.record_end(dmd.stream);
  profiler::Instance().kbytes_d2h += (sizeof(IdxType) +sizeof(ValType)) *nnz;
  call_back_data.ptr = this;
  call_back_data.seg1 = (dmd.gBuffer<<1) | dmd.local_buffer_id ;
  call_back_data.seg2 = dmd.cBid;
  
  CHK_CUDA_ERR(cudaLaunchHostFunc(dmd.stream, finish_data_d2h<Gs_t, Single>, &call_back_data));
}




} // namespace CESpGEMM
