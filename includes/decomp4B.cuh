#pragma once
#include"compress4B.h"
#include<type_traits>

static __device__ int warp_scan(int a, int laneId){
  constexpr unsigned int full_mask=0xffffffff;
  int tmp;
  tmp = __shfl_up_sync(full_mask, a, 1); a+=(laneId < 1 ? 0:tmp);
  tmp = __shfl_up_sync(full_mask, a, 2); a+=(laneId < 2 ? 0:tmp);
  tmp = __shfl_up_sync(full_mask, a, 4); a+=(laneId < 4 ? 0:tmp);
  tmp = __shfl_up_sync(full_mask, a, 8); a+=(laneId < 8 ? 0:tmp);
  tmp = __shfl_up_sync(full_mask, a, 16); a+=(laneId < 16 ? 0:tmp);
  return a;
}



template<int blocksize>
static __device__ int block_inclusive_scan(int val){
  constexpr int n_warps=blocksize/32;
  constexpr unsigned fullmask =0xffffffff;
  __shared__ int warp_sum[32];
  int laneId = threadIdx.x&31, warpId = threadIdx.x/32;
  val = warp_scan(val, laneId);
  if(laneId==31) warp_sum[warpId] = val;
  __syncthreads();
  //
  if(warpId == 0){
    int sum = warp_sum[laneId], tmp;
    #pragma unroll
    for(int i=1;i<n_warps;i<<=1){
      tmp = __shfl_up_sync(fullmask, sum, i);
      sum += (laneId < i ? 0: tmp);
    }
    // warp_sum[laneId] = warp_scan(sum, laneId);
    warp_sum[laneId] = sum;
  }
  __syncthreads();
  return val + (warpId>0?warp_sum[warpId-1] : 0);
}

__host__ __device__ __forceinline__ uint32_t cdiv(uint32_t x, uint32_t y){
  return (x+y-1)/y;
}

static size_t get_sharedsize_data_decomp(uint32_t block_size, uint32_t nz_head){
  size_t sData_units = cdiv(block_size * 4 + 6 * nz_head + 1, 4);
  size_t sControl_units = cdiv(block_size/(8/2) + 1, 4);
  size_t s_rel_pos_units = cdiv(nz_head * sizeof(uint16_t), 4);
  size_t s_length = nz_head;
  return (sData_units + sControl_units + s_rel_pos_units + s_length)*4;
}

template<int BLOCKSIZE, typename EIdType>
__global__ void decomp4B_knl(uint32_t nz_head, uint32_t *anc, uint8_t*data, uint8_t*control, EIdType*target){
  static_assert(BLOCKSIZE%128==0 && sizeof(EIdType)==4 && std::is_integral_v<EIdType>);
  extern __shared__ uint32_t s_buffer[];
  // __shared__ uint8_t sData[BLOCKSIZE*4+6*NZ_HEAD+1] ;
  // __shared__ uint8_t sControl[BLOCKSIZE/(8/2)+1];
  // __shared__ uint16_t s_rel_pos[NZ_HEAD];
  // __shared__ uint32_t s_length[NZ_HEAD];
      // 分配sData（uint8_t数组）
    size_t offset = 0;
    
    uint8_t* sData = (uint8_t*)(s_buffer + offset);
    size_t sData_units = cdiv(BLOCKSIZE*4+6*nz_head+1, 4);
    offset += sData_units;

    // 分配sControl（uint8_t数组）
    uint8_t* sControl = (uint8_t*)(s_buffer + offset);
    size_t sControl_units = cdiv(BLOCKSIZE/(8/2) + 1, 4);
    offset += sControl_units;

    // 分配s_rel_pos（uint16_t数组）
    uint16_t* s_rel_pos = (uint16_t*)(s_buffer + offset);
    size_t s_rel_pos_units = cdiv(nz_head * sizeof(uint16_t), 4);
    offset += s_rel_pos_units;

    // 分配s_length（uint32_t数组）
    uint32_t* s_length = s_buffer + offset;
    offset += nz_head;

  int segId = blockIdx.x;
  int elemId = threadIdx.x;
  int NzAnchor = GetPreNz4B(anc, segId);
  int start_byte = GetSrcPs4B(anc, segId), end_byte = GetSrcPs4B(anc, segId+1);
  int ValueAnchor = GetValue4B(anc, segId);

  int start_pos = segId * BLOCKSIZE;
  if(elemId < (BLOCKSIZE/4)){
    sControl[elemId + 1] = control[start_pos/4 + elemId];
  }else if(elemId == BLOCKSIZE-1){
    sControl[0]=0;
  }
  for(int i=0;i+start_byte<end_byte;i+=blockDim.x){
    if(elemId + i + start_byte < end_byte){
      sData[elemId + i] = data[elemId + i + start_byte];
    }
  }
  __syncthreads();
  uint8_t nzhead = sData[0];
  if(elemId < nzhead) {
    uint8_t a,b,c,d;
    int p=1+elemId*6;
    s_rel_pos[elemId] = sData[p]+sData[p+1]*256;
    a = sData[p+2];
    b = sData[p+3];
    c = sData[p+4];
    d = sData[p+5];
    s_length[elemId] = a+(b<<8)+(c<<16)+(d<<24);
  }
  __syncthreads();
  int bytes_of_elems = end_byte - start_byte - (1+nzhead*6);
  uint8_t *values = sData + (1+nzhead*6);
  int pcntl = (sControl[(elemId+3)/4] >> (((elemId-1)&3)*2)) & 3;
  int cntl = ( sControl[elemId/4+1] >> (((elemId ) &3)*2)) & 3;
  int read_pos = block_inclusive_scan<BLOCKSIZE>(pcntl) + elemId;
  // is that really right?
  int value = 0, w=1;
  #pragma unroll
  for(int i=0;i<=cntl;i++){
    value += w * values[read_pos + i];
    w *= 256;
  }
  value = block_inclusive_scan<BLOCKSIZE>(value); // value实际上是nnz的前缀和，不担心爆int
  int pref_nz = 0;
  #pragma unroll
  for(uint8_t i=0;i<nzhead;i++){
    int pos = s_rel_pos[i], len = s_length[i];
    pref_nz += pos<=elemId ? len:0;
    if(pos == elemId){
      target[pref_nz + NzAnchor + segId * BLOCKSIZE + elemId-1] = (EIdType)-len;
    }
  }
  if(read_pos < bytes_of_elems){
    target[pref_nz + NzAnchor + segId * BLOCKSIZE + elemId ] = value + ValueAnchor;
  }

}

