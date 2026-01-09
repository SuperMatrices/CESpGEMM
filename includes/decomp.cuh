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


template<typename T>
struct DynArray{
  T*_base;
  size_t n;
  __device__ __forceinline__ DynArray(void*base, size_t n):_base((T*)base), n(n){}
  __device__ __forceinline__ T& operator[](size_t idx){
    assert(idx<n);
    return _base[idx];
  }
} ;
struct DynSharedArrayAllocator
{
  uint32_t *_base;
  __device__ __forceinline__ 
  DynSharedArrayAllocator(uint32_t*base):_base(base){}

  template<typename T>
  __device__ __forceinline__ 
  void* allocPtr(size_t nElems){
    static_assert(sizeof(uint32_t)%sizeof(T)==0);
    void* ret = (void*)(_base);
    size_t offset = (nElems * sizeof(T) + 3)/4;
    _base += offset;
    return ret;
  }

  template<typename T>
  __device__ __forceinline__ 
  DynArray<T> allocArr(size_t nElems){
    return DynArray<T>(allocPtr<T>(nElems), nElems);
  }
} ;

inline void check_cuda_error(cudaError_t err, const char*s, const int lid){
  if(err!=cudaSuccess){
      fprintf(stderr,"CUDA error at %s %d: %s\n", s, lid, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

}

#define CHK_CUDA(call) check_cuda_error(call, __FILE__, __LINE__)

using AnchorData_t = uint32_t;




template<int BLOCKSIZE, typename EIdType>
__global__ void decomp_knl(uint32_t nz_head, AnchorData_t *anc, uint8_t*data, uint8_t*control, EIdType*target);


// template<typename T>
// void decompress_cpu(int nAll, int nNzValues, int nBytes, int nSegs, int SegLen, Anchor*anc, uint8_t *data, uint8_t *control, T*target);

template<typename T>
void decompress_gpu(int nAll, int nNzValues, int nBytes, int nSegs, int SegLen, unsigned*anc, uint8_t *data, uint8_t *control, T*target);

template<typename T>
void decompress_cpu_with_zero_per_seg(int nAll, int nNzValues, int nBytes, int nSeg, int SegLen, unsigned* anc, uint8_t* data, uint8_t *control, T*target);

template<typename EIdType>
void decomp_launch(AnchorData_t *anc, uint8_t*data, uint8_t*control, EIdType *target, const int seglen, const int nz_head, int num_blocks, cudaStream_t stream);
