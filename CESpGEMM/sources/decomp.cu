#include"decomp.cuh"
#include<cuda_fp16.h>
#include<chrono>
#include<type_traits>
#include"generator.h"

// template void decompress_gpu(int,int,int,int,int,unsigned*,uint8_t*, uint8_t*t,int*);

// template<typename T>
// void decompress_cpu(int nAll, int nNzValues, int nBytes, int nSegs, int SegLen, Anchor*anc, uint8_t *data, uint8_t *control, T *target){
//   static uint16_t position[4];
//   static uint32_t zero_len[4];
  
//   for(int i=0;i<nSegs-1;i++){
//     auto [anc_val, anc_zero, start_byte] = anc[i];
//     auto [anc_val2, anc_zero2, end_byte] = anc[i+1];
//     uint8_t nz = data[start_byte++];
//     if(nz){
//       printf("nz=%d, in segid=%d, byteid=%d\n", nz, i, start_byte);
//     }
//     for(int j=0;j<nz;j++){
//       uint16_t pos = data[start_byte] + (data[start_byte+1] << 8);
//       start_byte += 2;
//       uint32_t val = data[start_byte + 3]; 
//       val = (val<<8) + data[start_byte+2];
//       val = (val<<8) + data[start_byte+1];
//       val = (val<<8) + data[start_byte];
//       position[j] = pos;
//       printf("pos = %d\n", pos);
//       zero_len[j] = val;
//       start_byte += 4;
//     }
//     uint8_t nzhead_id = 0;
//     uint32_t seg_data_sum=0;
//     uint32_t nzhead_sum = 0;

//     int nnz_id_start = i * SegLen;
//     int nnz_id_end = min(nnz_id_start + SegLen, nNzValues);
//     int nnz = nnz_id_end - nnz_id_start;

//     for(int j=0;j<nnz;j++){
//       int basePos = j + (anc_zero+nnz_id_start) + nzhead_sum;
//       if(nzhead_id < nz && j>= position[nzhead_id]){
//         // printf("seg nz value %d to [%d, %d]\n", seg_data_sum + anc_val, basePos, basePos+zero_len[nzhead_id]-1);
//         for(int k=basePos;k<basePos+zero_len[nzhead_id];k++){
//           target[k] = seg_data_sum + anc_val;
//         }
//         basePos += zero_len[nzhead_id];
//         nzhead_sum += zero_len[nzhead_id ++];
//       }
//       seg_data_sum += data[start_byte++];
//       int nnz_idj = nnz_id_start + j;
      
//       if(control[nnz_idj/8] & (1<<(nnz_idj&7))){
//         seg_data_sum += data[start_byte++] << 8;
//       }
//       target[basePos] = static_cast<T>(seg_data_sum + anc_val);
//     }
//   }
// }

namespace{
  void wrong_info(int pos, int expected, int got){
    printf("wrong at %d, expected %d, got %d\n", pos, expected, got);
  }
  template<typename T>
  bool check(int len, const T*origin_ptr, const T*result){
    int bad = 0;
    auto handle_err = [&](int pos, int expected, int got) {
      wrong_info(pos, expected, got);
      bad ++ ;
    } ;
    for(int i=0;i<len;i++){
      if((int)result[i]==-1){
        if(origin_ptr[i]-origin_ptr[i-1] != 0){
          printf("surrounding ptr: i-1:%d, i:%d, i+1:%d\n", origin_ptr[i-1], origin_ptr[i], origin_ptr[i+1]);
          handle_err(i, origin_ptr[i], result[i]);
        }
      }
      else if((int)result[i] < 0){
        int l = - result[i];
        int v = i>=l ? result[i-l] : 0;
        // printf(">--- at the end of zeroseg, pos=%d, length=%d, value=%d\n", i, l, v);
        // printf("surrounding ptr: i-1:%d, i:%d, i+1:%d ----<\n", origin_ptr[i-1], origin_ptr[i], origin_ptr[i+1]);
        if(v != origin_ptr[i] ){
          handle_err(i, origin_ptr[i], v);
        }
      }
      else{
        if(origin_ptr[i]!=result[i]){
          handle_err(i, origin_ptr[i], result[i]);
        }
      }
    }
    if(bad) return false;
    return true;
  }
}

namespace CESpGEMM{
template
bool validate_decompress_on_gpu(const csr<uint32_t,float> &c, const compress_t &cmp);

template<typename EIdType, typename ValType>
bool validate_decompress_on_gpu(const csr<EIdType, ValType> &c, const compress_t &cmp){
  uint32_t *d_anchors;
  uint8_t* d_src, *d_control;
  uint32_t *d_target;
  int nSegs = cmp.num_segs;
  int nBytes = cmp.bytes_of_data;
  int nNzValues = cmp.num_values;
  int nAll = c.nr + 1;
  EIdType*target = new EIdType[nAll];


  CHK_CUDA(cudaMalloc((void**)&d_anchors, sizeof(int) * 3 * (nSegs+1)));
  CHK_CUDA(cudaMalloc((void**)&d_src, sizeof(uint8_t) * (nBytes+512)));
  CHK_CUDA(cudaMalloc((void**)&d_control, sizeof(uint8_t) * (nNzValues+64)));
  CHK_CUDA(cudaMalloc((void**)&d_target, sizeof(int) * nAll));
  CHK_CUDA(cudaMemset(d_target, -1, sizeof(uint32_t) * nAll));

  CHK_CUDA(cudaMemcpy(d_anchors, cmp.anchor_data, sizeof(int)* 3 *(nSegs+1), cudaMemcpyHostToDevice));
  CHK_CUDA(cudaMemcpy(d_src, cmp.data, sizeof(uint8_t) * nBytes, cudaMemcpyHostToDevice));
  CHK_CUDA(cudaMemcpy(d_control, cmp.control, sizeof(uint8_t) * cmp.nbytes_control, cudaMemcpyHostToDevice));
  decomp_knl<512><<<nSegs, 512>>>(
    d_anchors, d_src, d_control, d_target
  );
  CHK_CUDA(cudaDeviceSynchronize());
  CHK_CUDA(cudaGetLastError());
  CHK_CUDA(cudaMemcpy(target, d_target, sizeof(EIdType) * nAll, cudaMemcpyDeviceToHost));
  bool result = check(nAll, c.ptr, target);
  delete[]target;
  return result;
}



}


__device__ int warp_scan(int a, int laneId){
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
__device__ int block_inclusive_scan(int val){
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

#define BlockSinglePrint if(threadIdx.x==0)printf

template
__global__ void decomp_knl<512, uint32_t>(AnchorData_t *anc, uint8_t*data, uint8_t*control, uint32_t *target);
template
__global__ void decomp_knl<512, int>(AnchorData_t *anc, uint8_t*data, uint8_t*control, int *target);

template<int blocksize, typename EIdType>
__global__ void decomp_knl(AnchorData_t *anc, uint8_t*data, uint8_t*control, EIdType *target){
  static_assert(blocksize==512);
  static_assert(std::is_integral_v<EIdType>);
  static_assert(sizeof(EIdType) == 4);
 
  __shared__ uint8_t sData[512*2+6*4+1]; //max: all are 2bytes: 512*2+6*4+1 bytes
  __shared__ uint8_t sControl[65];
  __shared__ uint16_t s_rel_pos[4];
  __shared__ uint32_t s_length[4];

  // 读的时候往右移一位，sControl[0]设成0, 对读取到的control求前缀和，再加上tid
  // Anchor数组改成int4数组
  // value 在原始数组补0
  // 先用shuffle，先只做前缀和。
  
  int segId = blockIdx.x;
  int elemId = threadIdx.x;
  int NzAnchor = GetPreNz(anc, segId);
  int start_byte = GetSrcPs(anc, segId), end_byte= GetSrcPs(anc, segId+1);
  int ValueAnchor = GetValue(anc, segId);
  int start_pos = segId * 512;
  if(elemId<64){
    sControl[elemId+1] = control[start_pos/8 + elemId];
  }else if(elemId == blocksize-1){
    sControl[0] = 0;
  }
  for(int i=0;i+start_byte<end_byte;i+=blockDim.x){
    if(elemId + i + start_byte < end_byte){ // keep the "if"
      sData[elemId + i] = data[elemId + i + start_byte];
    }
  }
  
  __syncthreads();
  uint8_t nzhead = sData[0];
  if(elemId < nzhead){
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

  // if(segId == 0){
  //   for(int i=0;i<nzhead;i++){
  //     BlockSinglePrint("relPos=%d, length=%d\n", s_rel_pos[i], s_length[i]);
  //   }
  // }
  int bytes_of_elems = end_byte-start_byte-(1+nzhead*6);
  uint8_t* values = sData+(1+nzhead*6);
  int pcntl = ( sControl[(elemId+7)/8] >> ((elemId-1) & 7) ) &1; // find the previous cnt value , to get prefixsum to locate
  int cntl = ( (sControl[elemId/8+1])>>(elemId & 7) ) &1; 
  int read_pos = block_inclusive_scan<blocksize>(pcntl) + elemId;
  int value = values[read_pos] + (cntl ? values[read_pos+1]:0)*256;
  // if(segId==10){
  //   printf("gpu: %d:%d,%d\n", elemId, read_pos, value);
  // }
  value = block_inclusive_scan<blocksize>(value);
  int pref_nz = 0;
  #pragma unroll
  for(uint8_t i=0;i<nzhead;i++){
    int pos = s_rel_pos[i], len = s_length[i];
    pref_nz += pos<=elemId ? len:0;
    if(pos == elemId) {
      target[pref_nz + NzAnchor + segId * 512 + elemId - 1] = (EIdType)-len;
    }
  }


  // {
  //   BlockScan_t(temp_storage).ExclusiveSum(pref_nz, pref_nz);
  // }
  // if(segId <= 10){
  //   printf("segId=%d, elemId=%d, pcntl=%d, read_pos=%d, pref_nz=%d, NzAnchor=%d, targetpos=%d, value=%d\n", segId, elemId, pcntl, read_pos, pref_nz, NzAnchor, pref_nz + NzAnchor + segId * 512 + elemId, value);
  // }
  if(read_pos < bytes_of_elems){
    target[pref_nz + NzAnchor + segId * 512 + elemId] = value + ValueAnchor;
  }
}


template<typename T>
void decompress_gpu(int nAll, int nNzValues, int nBytes, int nSegs, int SegLen, unsigned*anc, uint8_t *data, uint8_t *control, T *target){
  using namespace std::chrono;
	static_assert(sizeof(T)==4);
	AnchorData_t *d_anchors;

  uint8_t* d_src, *d_control;
  int *d_target;
  
    CHK_CUDA(cudaMalloc((void**)&d_anchors, sizeof(AnchorData_t) * 3 * (nSegs+1)));
    CHK_CUDA(cudaMalloc((void**)&d_src, sizeof(uint8_t) * (nBytes+512)));
    CHK_CUDA(cudaMalloc((void**)&d_control, sizeof(uint8_t) * (nNzValues+64)));
    CHK_CUDA(cudaMalloc((void**)&d_target, sizeof(int) * nAll));
    CHK_CUDA(cudaMemset(d_target, -1, sizeof(uint32_t) * nAll));

      CHK_CUDA(cudaMemcpy(d_anchors, anc, sizeof(AnchorData_t)* 3 *(nSegs+1), cudaMemcpyHostToDevice));
      CHK_CUDA(cudaMemcpy(d_src, data, sizeof(uint8_t) * nBytes, cudaMemcpyHostToDevice));
      CHK_CUDA(cudaMemcpy(d_control, control, sizeof(uint8_t) * nNzValues, cudaMemcpyHostToDevice));

  // fprintf(stderr, "nsegs=%d\n", nSegs);
  
  auto t0 = steady_clock::now();
  decomp_knl<512, int><<<nSegs, 512>>>(d_anchors, d_src, d_control, d_target);
  CHK_CUDA(cudaDeviceSynchronize());
  CHK_CUDA(cudaGetLastError());
  auto t1 = steady_clock::now();
  
  CHK_CUDA(cudaMemcpy(target, d_target, sizeof(T) * nAll, cudaMemcpyDeviceToHost));
  auto t2 = steady_clock::now();
}

template
void decompress_cpu_with_zero_per_seg(int nAll, int nNzValues, int nBytes, int nSeg, int SegLen, unsigned* anc, uint8_t* data, uint8_t *control, int*target);


template<typename T>
void decompress_cpu_with_zero_per_seg(int nAll, int nNzValues, int nBytes, int nSegs, int SegLen, unsigned* anc, uint8_t* data, uint8_t *control, T*target){
  for(int sid=0;sid<nSegs;sid++){
    // anc.pre
    // uint32_t last_nZ_offset = sid * SegLen;
    uint32_t start_byte = GetSrcPs(anc, sid);
    uint32_t end_byte = GetSrcPs(anc, sid+1);
    printf("seg%d: start_byte=%d, end_byte=%d\n", sid, start_byte, end_byte);
    uint32_t pref_nz = GetPreNz(anc, sid);
    uint32_t psum_value = GetValue(anc, sid) ;
    uint8_t nz = data[start_byte++];
    uint32_t relpos[4];
    uint32_t zlength[4];
    printf("nz=%d\n", nz);

    
    for(uint8_t zid=0;zid<nz;zid++){
      uint8_t p0 = data[start_byte++];
      uint8_t p1 = data[start_byte++]; 
      relpos[zid] = p0 + (p1*256);
      uint8_t l0 = data[start_byte ++ ] << (8*0) ;
      uint8_t l1 = data[start_byte ++ ] << (8*1) ;
      uint8_t l2 = data[start_byte ++ ] << (8*2) ;
      uint8_t l3 = data[start_byte ++ ] << (8*3) ;
      zlength[zid] = l0 + l1 + l2 + l3;
      printf("relpos=%d, zlength=%d\n", relpos[zid], zlength[zid]);
    }

    for(uint32_t value_pos = sid*SegLen, relp=0, nzp=0; start_byte<end_byte; value_pos++, relp++){
      while(nzp< nz && relpos[nzp] <= relp){
        pref_nz += zlength[nzp];
        if(relpos[nzp] == relp) {
          target[value_pos + pref_nz - 1] = -zlength[nzp];
        }
        nzp ++ ;
      }

      
      bool ctrl = control[value_pos/8] & (1<<(value_pos & 7));
      
      uint32_t value = data[start_byte ++];
      psum_value += value;

      if(ctrl){
        psum_value += data[start_byte ++] * 256;
      }
      assert(value_pos + pref_nz < nAll);
      target[value_pos + pref_nz] = psum_value;
    }
  }

}
