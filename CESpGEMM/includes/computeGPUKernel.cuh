#include"CSR.h"

namespace CESpGEMM
{

template<typename EIdType>
__device__ __forceinline__ int process_len(int pos, EIdType *ptr_arr, int this_ptr,int next_ptr){
  if(next_ptr < 0) return -1;
  if(this_ptr < 0){
    // if(pos == -1){
    //   printf("bid = pos=%d, this_ptr=%d, next_ptr=%d\n", pos, this_ptr, next_ptr);
    // }
    pos += this_ptr;
    return (pos>=0?ptr_arr[pos]:0);
  }
  return this_ptr;
}


template<typename EIdType>
__global__ void count_flops(
    EIdType start_ptr,
    EIdType *__restrict__ ptrA, IdxType *__restrict__  idxA, EIdType *__restrict__ ptrB,
    EIdType *__restrict__ flop
);

template<typename EIdType, typename ValType>
__global__ void expand_dense_h2d(
    EIdType start_ptr,
    EIdType *__restrict__ ptrA, EIdType *__restrict__ ptrB, 
    IdxType *__restrict__ sub_idxA, IdxType *__restrict__ idxB,
    ValType *__restrict__ sub_valA, ValType *__restrict__ valB,
    EIdType *__restrict__ inner_offset,
    IdxType *__restrict__ tmp_idx,
    ValType *__restrict__ tmp_val
);

template<typename EIdType, typename ValType>
__global__ void merge_interm_result(
    IdxType block_nr, IdxType block_nc, IdxType c_start, 
    EIdType *__restrict__ inner_offset,
    IdxType *__restrict__ tmp_idx /*in*/, ValType *__restrict__ tmp_val, /*in*/
    IdxType *__restrict__ out_idx, ValType *__restrict__ out_val, 
    IdxType *__restrict__ nnzC
);

template<typename EIdType, typename ValType>
__global__ void collect(
    EIdType *__restrict__ flop_offset, EIdType *__restrict__ nnz_offset, 
    IdxType *__restrict__ idx_in, ValType *__restrict__ val_in,
    IdxType *__restrict__ idx_out, ValType *__restrict__ val_out
);






//---------------------------above are ordinary kernels--------------------
//---------------------------below are kernels with compress

template<typename SrcEType, typename EIdType>
__global__ void count_flops_comp(
    SrcEType start_ptr,
    SrcEType *__restrict__ ptrA, IdxType *__restrict__  idxA, EIdType *__restrict__ ptrB,
    EIdType *__restrict__ flop
);

template<typename SrcEType, typename EIdType, typename ValType>
__global__ void expand_dense_h2d_comp(
    SrcEType start_ptr,
    SrcEType *__restrict__ ptrA, EIdType *__restrict__ ptrB, 
    IdxType *__restrict__ sub_idxA, IdxType *__restrict__ idxB,
    ValType *__restrict__ sub_valA, ValType *__restrict__ valB,
    EIdType *__restrict__ inner_offset,
    IdxType *__restrict__ tmp_idx,
    ValType *__restrict__ tmp_val
);


template<typename EIdType, typename ValType>
__global__ void merge_interm_result_glb_acum(
    IdxType block_nr, IdxType block_nc, IdxType c_start, 
    EIdType *__restrict__ inner_offset,
    IdxType *__restrict__ tmp_idx /*in*/, ValType *__restrict__ tmp_val, /*in*/
    ValType *__restrict__ accum,
    IdxType *__restrict__ out_idx, ValType *__restrict__ out_val, 
    IdxType *__restrict__ nnzC
);

//above are kernels with compress


template<size_t D>
struct GDType{
  using Tp = IdxType;
} ;
template<>
struct GDType<8>{
  using Tp = ull;
} ;




template<typename EIdType>
__global__ void count_flops(
  EIdType start_ptr,
  EIdType *__restrict__ ptrA, IdxType *__restrict__  idxA, EIdType *__restrict__ ptrB,
  EIdType *__restrict__ flop
){
  int bid = blockIdx.x;
  __shared__ EIdType tot;
  if(threadIdx.x == 0){tot=0;}
  __syncthreads();
  EIdType partial=0;
  for(IdxType i=ptrA[bid] + threadIdx.x;i<ptrA[bid+1]; i+=blockDim.x){
    IdxType cid = idxA[i - start_ptr];
    // if(bid == 0){
    //   printf("IN COUNT: r%d, t%d, pa%d, col%d, LB%d, RB%d\n", bid, threadIdx.x, i, cid, ptrB[cid], ptrB[cid+1]);
    // }

    partial += ptrB[cid+1]-ptrB[cid];
  }
  atomicAdd(&tot, partial);
  __syncthreads();
  if(threadIdx.x==0){
    flop[bid] = tot;
  }
}

template<typename EIdType, typename ValType>
__global__ void expand_dense_h2d(
    EIdType start_ptr,
    EIdType *__restrict__ ptrA, EIdType *__restrict__ ptrB, 
    IdxType *__restrict__ sub_idxA, IdxType *__restrict__ idxB,
    ValType *__restrict__ sub_valA, ValType *__restrict__ valB,
    EIdType *__restrict__ inner_offset,
    IdxType *__restrict__ tmp_idx,
    ValType *__restrict__ tmp_val
  )
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	__shared__ EIdType shared_offset;
	EIdType idx1;
	if(threadIdx.x == 0){
		shared_offset = inner_offset[bid];
	}
	__syncthreads();
	for(EIdType pa = ptrA[bid] + tid; pa < ptrA[bid+1]; pa += blockDim.x){
		IdxType col = sub_idxA[pa - start_ptr];
		ValType mul = sub_valA[pa - start_ptr];
		EIdType idx2 = ptrB[col];
		int lenb = ptrB[col+1] - ptrB[col];
		idx1 = atomicAdd(&shared_offset, (EIdType) lenb);
    // if( bid == 0 ){
    //   printf("r%d, t%d, pa%d, col%d, LB%d, RB%d, len%d, j%d, idx1=%d, idx2=%d, tmpidx%d, tmpval=%.2f, %p\n", bid, tid, pa, col, ptrB[col], ptrB[col+1], lenb, 0, idx1, idx2, idxB[idx2+0+0], valB[idx2+0+0], idxB);
    // }
		int j=0;
		for(j=0;j+7<lenb;j+=8){
			tmp_idx[idx1 + j + 0] = idxB[idx2 + j + 0];
			tmp_idx[idx1 + j + 1] = idxB[idx2 + j + 1];
			tmp_idx[idx1 + j + 2] = idxB[idx2 + j + 2];
			tmp_idx[idx1 + j + 3] = idxB[idx2 + j + 3];
			tmp_idx[idx1 + j + 4] = idxB[idx2 + j + 4];
			tmp_idx[idx1 + j + 5] = idxB[idx2 + j + 5];
			tmp_idx[idx1 + j + 6] = idxB[idx2 + j + 6];
			tmp_idx[idx1 + j + 7] = idxB[idx2 + j + 7];

			tmp_val[idx1 + j + 0] = mul*valB[idx2 + j + 0];
			tmp_val[idx1 + j + 1] = mul*valB[idx2 + j + 1];
			tmp_val[idx1 + j + 2] = mul*valB[idx2 + j + 2];
			tmp_val[idx1 + j + 3] = mul*valB[idx2 + j + 3];
			tmp_val[idx1 + j + 4] = mul*valB[idx2 + j + 4];
			tmp_val[idx1 + j + 5] = mul*valB[idx2 + j + 5];
			tmp_val[idx1 + j + 6] = mul*valB[idx2 + j + 6];
			tmp_val[idx1 + j + 7] = mul*valB[idx2 + j + 7];
		}

		if(j+3<lenb){
			tmp_idx[idx1 + j + 0] = idxB[idx2 + j + 0];
			tmp_idx[idx1 + j + 1] = idxB[idx2 + j + 1];
			tmp_idx[idx1 + j + 2] = idxB[idx2 + j + 2];
			tmp_idx[idx1 + j + 3] = idxB[idx2 + j + 3];
			
			tmp_val[idx1 + j + 0] = mul*valB[idx2 + j + 0];
			tmp_val[idx1 + j + 1] = mul*valB[idx2 + j + 1];
			tmp_val[idx1 + j + 2] = mul*valB[idx2 + j + 2];
			tmp_val[idx1 + j + 3] = mul*valB[idx2 + j + 3];
			
			j+=4;
		}

		if(j+1<lenb){
			tmp_idx[idx1 + j + 0] = idxB[idx2 + j + 0];
			tmp_idx[idx1 + j + 1] = idxB[idx2 + j + 1];
			tmp_val[idx1 + j + 0] = mul*valB[idx2 + j + 0];
			tmp_val[idx1 + j + 1] = mul*valB[idx2 + j + 1];
			j+=2;
		}
		if(j<lenb){
			tmp_idx[idx1 + j] = idxB[idx2 + j];
			tmp_val[idx1 + j] = mul*valB[idx2 + j];
		}

    // if( bid == 0 ){
    //   printf("AFTER: r%d, t%d, pa%d, col%d, len%d, j%d, idx1=%d, idx2=%d, tmpidx%d, tmpval=%.2f, tmpidxget%d, %p\n", bid, tid, pa, col, lenb, 0, idx1, idx2, idxB[idx2+0+0], valB[idx2+0+0], tmp_idx[idx1+0], idxB);
    // }

	}
}


template<typename EIdType, typename ValType>
__global__ void merge_interm_result(
    IdxType block_nr, IdxType block_nc, IdxType c_start, 
    EIdType *__restrict__ inner_offset,
    IdxType *__restrict__ tmp_idx /*in*/, ValType *__restrict__ tmp_val, /*in*/
    IdxType *__restrict__ out_idx, ValType *__restrict__ out_val, 
    IdxType *__restrict__ nnzC
)
{
  using UValType = typename GDType<sizeof(ValType)>::Tp;
  
  __shared__ EIdType shared_nnz;
  __shared__ ValType dense_val[4000];
  
  int bid = blockIdx.x;
  for(IdxType i=threadIdx.x;i<block_nc;i+=blockDim.x){
    dense_val [i] = 0;
  }
  if(threadIdx.x == blockDim.x-1) shared_nnz = 0;
  
  EIdType baseTmp = inner_offset[bid], endOffset = inner_offset[bid+1];
  // if(threadIdx.x == blockDim.x-1) printf("%d:%d,%d\n", bid, baseTmp, endOffset);

  __syncthreads();

  for(EIdType i=baseTmp + threadIdx.x; i<endOffset; i+=blockDim.x){
    IdxType col = tmp_idx[i];
    ValType val = tmp_val[i];
    assert(col < block_nc);
    ValType old_val = atomicAdd(dense_val + (col) , val);
    // ValType old_val = 0;

    if( *(UValType*)(&old_val) == (UValType)0){
      EIdType oldNNZ = atomicAdd(&shared_nnz, 1);
      out_idx[baseTmp + oldNNZ] = col;
    }
  }
  __syncthreads();
  for(EIdType i=threadIdx.x; i<shared_nnz; i+=blockDim.x){
    IdxType col = out_idx[baseTmp + i];
    out_val[baseTmp + i] = dense_val[col];
  }
  if(threadIdx.x == blockDim.x-1) nnzC[bid] = shared_nnz;
}

template<typename EIdType, typename ValType>
__global__ void collect(
    EIdType *__restrict__ flop_offset, EIdType *__restrict__ nnz_offset, 
    IdxType *__restrict__ idx_in, ValType *__restrict__ val_in,
    IdxType *__restrict__ idx_out, ValType *__restrict__ val_out
){
  int bid = blockIdx.x;
  EIdType baseIn = flop_offset[bid], baseOut = nnz_offset[bid];
  EIdType nnz = nnz_offset[bid+1] - baseOut;
  
  for(EIdType i=0;i<nnz;i+=blockDim.x){
    idx_out[baseOut + i] = idx_in[baseIn + i];
    val_out[baseOut + i] = val_in[baseIn + i];
  }
}

//=======================below are implementation for comp=======================================

template<typename SrcEType, typename EIdType>
__global__ void count_flops_comp(
    SrcEType start_ptr,
    SrcEType *__restrict__ ptrA, IdxType *__restrict__  idxA, EIdType *__restrict__ ptrB,
    EIdType *__restrict__ flop
){
  int bid = blockIdx.x;
  __shared__ EIdType tot;
  if(threadIdx.x == 0){tot=0;}
  __syncthreads();
  EIdType partial=0;
  IdxType end_idxA = ptrA[bid+1] - start_ptr;
  for(IdxType i=ptrA[bid] + threadIdx.x - start_ptr;i<end_idxA; i+=blockDim.x){
    IdxType cid = idxA[i];
    EIdType npv = ptrB[cid+1], pv = process_len(cid, ptrB, ptrB[cid], npv);
    partial += (pv == (EIdType)-1) ? 0 : npv-pv;
    // if(npv < pv){
    //   printf("what?!:cid=%d, npv=%d, pv=%d, ppv=%d\n", cid, npv, pv, ptrB[cid]);
    // }
  }
  atomicAdd(&tot, partial);
  __syncthreads();
  if(threadIdx.x==0){
    flop[bid] = tot;
  }
}

template<typename SrcEIdType, typename EIdType, typename ValType>
__global__ void expand_dense_h2d_comp(
    SrcEIdType start_ptr,
    SrcEIdType *__restrict__ ptrA, EIdType *__restrict__ ptrB, 
    IdxType *__restrict__ sub_idxA, IdxType *__restrict__ idxB,
    ValType *__restrict__ sub_valA, ValType *__restrict__ valB,
    EIdType *__restrict__ inner_offset,
    IdxType *__restrict__ tmp_idx,
    ValType *__restrict__ tmp_val
)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	__shared__ EIdType shared_offset;
	EIdType idx1;
	if(threadIdx.x == 0){
		shared_offset = inner_offset[bid];
	}
	__syncthreads();
  IdxType paEnd = ptrA[bid+1] - start_ptr;
	for(IdxType pa = ptrA[bid] + tid - start_ptr; pa < paEnd; pa += blockDim.x){
		IdxType col = sub_idxA[pa];
		ValType mul = sub_valA[pa];
    EIdType npv = ptrB[col+1], pv = process_len(col, ptrB, ptrB[col], npv);
    // printf("col=%d, ppv=%d, npv=%d, pv=%d\n", col, ppv, npv, pv);

		EIdType idx2 = pv;
		int lenb = (pv==(EIdType)-1) ? 0 : (npv - pv);
		idx1 = atomicAdd(&shared_offset, (EIdType) lenb);

		int j=0;
		for(j=0;j+7<lenb;j+=8){
			tmp_idx[idx1 + j + 0] = idxB[idx2 + j + 0];
			tmp_idx[idx1 + j + 1] = idxB[idx2 + j + 1];
			tmp_idx[idx1 + j + 2] = idxB[idx2 + j + 2];
			tmp_idx[idx1 + j + 3] = idxB[idx2 + j + 3];
			tmp_idx[idx1 + j + 4] = idxB[idx2 + j + 4];
			tmp_idx[idx1 + j + 5] = idxB[idx2 + j + 5];
			tmp_idx[idx1 + j + 6] = idxB[idx2 + j + 6];
			tmp_idx[idx1 + j + 7] = idxB[idx2 + j + 7];

			tmp_val[idx1 + j + 0] = mul*valB[idx2 + j + 0];
			tmp_val[idx1 + j + 1] = mul*valB[idx2 + j + 1];
			tmp_val[idx1 + j + 2] = mul*valB[idx2 + j + 2];
			tmp_val[idx1 + j + 3] = mul*valB[idx2 + j + 3];
			tmp_val[idx1 + j + 4] = mul*valB[idx2 + j + 4];
			tmp_val[idx1 + j + 5] = mul*valB[idx2 + j + 5];
			tmp_val[idx1 + j + 6] = mul*valB[idx2 + j + 6];
			tmp_val[idx1 + j + 7] = mul*valB[idx2 + j + 7];
		}

		if(j+3<lenb){
			tmp_idx[idx1 + j + 0] = idxB[idx2 + j + 0];
			tmp_idx[idx1 + j + 1] = idxB[idx2 + j + 1];
			tmp_idx[idx1 + j + 2] = idxB[idx2 + j + 2];
			tmp_idx[idx1 + j + 3] = idxB[idx2 + j + 3];
			
			tmp_val[idx1 + j + 0] = mul*valB[idx2 + j + 0];
			tmp_val[idx1 + j + 1] = mul*valB[idx2 + j + 1];
			tmp_val[idx1 + j + 2] = mul*valB[idx2 + j + 2];
			tmp_val[idx1 + j + 3] = mul*valB[idx2 + j + 3];
			
			j+=4;
		}

		if(j+1<lenb){
			tmp_idx[idx1 + j + 0] = idxB[idx2 + j + 0];
			tmp_idx[idx1 + j + 1] = idxB[idx2 + j + 1];
			tmp_val[idx1 + j + 0] = mul*valB[idx2 + j + 0];
			tmp_val[idx1 + j + 1] = mul*valB[idx2 + j + 1];
			j+=2;
		}
		if(j<lenb){
			tmp_idx[idx1 + j] = idxB[idx2 + j];
			tmp_val[idx1 + j] = mul*valB[idx2 + j];
		}

    // if( bid == 0 ){
    //   printf("AFTER: r%d, t%d, pa%d, col%d, len%d, j%d, idx1=%d, idx2=%d, tmpidx%d, tmpval=%.2f, tmpidxget%d, %p\n", bid, tid, pa, col, lenb, 0, idx1, idx2, idxB[idx2+0+0], valB[idx2+0+0], tmp_idx[idx1+0], idxB);
    // }

	}
  // __syncthreads();
  // if(threadIdx.x == 0){
  //   if(shared_offset != inner_offset[bid + 1]){
  //     printf("bid = %d, shared offset = %d, inner offset=%d\n", bid, shared_offset, inner_offset[bid+1]);
  //   }
  // }
}

template<typename EIdType, typename ValType>
__global__ void merge_interm_result_glb_acum(
    IdxType block_nr, IdxType block_nc, IdxType c_start, 
    EIdType *__restrict__ inner_offset,
    IdxType *__restrict__ tmp_idx /*in*/, ValType *__restrict__ tmp_val, /*in*/
    ValType *__restrict__ accum,
    IdxType *__restrict__ out_idx, ValType *__restrict__ out_val, 
    IdxType *__restrict__ nnzC
){
  using UValType = typename GDType<sizeof(ValType)>::Tp;
  
  __shared__ EIdType shared_nnz;
  int bid = blockIdx.x;
  ValType* dense_val = accum + block_nc * bid;

  if(threadIdx.x == blockDim.x-1) shared_nnz = 0;

  __syncthreads();
  for(int iter=0;iter<=(block_nr/gridDim.x);iter++){
    int rel_rid = iter * gridDim.x + bid;
    if(rel_rid >= block_nr) break;
    EIdType baseTmp = inner_offset[rel_rid], endOffset = inner_offset[rel_rid+1];
    for(EIdType i=baseTmp + threadIdx.x; i<endOffset; i+= blockDim.x){
      IdxType col = tmp_idx[i];
      ValType val = tmp_val[i];

      // if(col>=block_nc){
      //   printf("threadIdx=%d, i=%d, col=%d, val=%.3f, blocknc=%d\n", threadIdx.x, i, col, val, block_nc);
      // }
      assert(col<block_nc);

      ValType old_val = atomicAdd(dense_val + (col) , val);
      // ValType old_val = 0;
      if( *(UValType*)(&old_val) == (UValType)0){
        EIdType oldNNZ = atomicAdd(&shared_nnz, 1);
        out_idx[baseTmp + oldNNZ] = col;
      }
    }
    __syncthreads();
    for(EIdType i = threadIdx.x; i<shared_nnz; i += blockDim.x){
      IdxType col = out_idx[baseTmp + i];
      out_val[baseTmp + i] = dense_val[col];
      dense_val[col] = 0;
    }
    __syncthreads();
    if(threadIdx.x == blockDim.x-1){
      nnzC[rel_rid] = shared_nnz;
      shared_nnz = 0;
    }
    __syncthreads();
  }

}


}