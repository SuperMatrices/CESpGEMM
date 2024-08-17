#pragma once
#include<vector>
#include<cstdint>
#include<cuda.h>
#include<cuda_runtime.h> 
#include"compressor.h"
enum DecompCode{
  DECOMP_SUCCESS=0,
  DECOMP_EXCEED=1,
  DECOMP_ERROR=2
} ;


inline void check_cuda_error(cudaError_t err, const char*s, const int lid){
  if(err!=cudaSuccess){
      fprintf(stderr,"CUDA error at %s %d: %s\n", s, lid, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

}

#define CHK_CUDA(call) check_cuda_error(call, __FILE__, __LINE__)

using AnchorData_t = uint32_t;




template<int blocksize, typename EIdType>
__global__ void decomp_knl(AnchorData_t *anc, uint8_t*data, uint8_t*control, EIdType*target);


// template<typename T>
// void decompress_cpu(int nAll, int nNzValues, int nBytes, int nSegs, int SegLen, Anchor*anc, uint8_t *data, uint8_t *control, T*target);

template<typename T>
void decompress_gpu(int nAll, int nNzValues, int nBytes, int nSegs, int SegLen, unsigned*anc, uint8_t *data, uint8_t *control, T*target);

template<typename T>
void decompress_cpu_with_zero_per_seg(int nAll, int nNzValues, int nBytes, int nSeg, int SegLen, unsigned* anc, uint8_t* data, uint8_t *control, T*target);
