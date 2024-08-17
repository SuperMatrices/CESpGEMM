#pragma once

#include"CSR.h"
#include<vector>
#include<atomic>
#include"compressor.h"

namespace CESpGEMM
{


template<typename SrcEType, typename EIdType, typename ValType>
class GlobalStorage 
{
public:
  using srcEtype = SrcEType;
  using eidType = EIdType;
  using valType = ValType;

  size_t pool_size;
  int num_workers;
  // ValType* cpu_dense;
  IdxType blockSizeA, blockSizeB, numBlocksA, numBlocksB;

  csr<SrcEType, ValType> csrA;
  csc<SrcEType, ValType> csrB_T;
  ull *csra_idx_64;
  ull *cscb_idx_64;
  // std::vector<csr<EIdType, ValType>> vcsrB;
  std::vector< std::unique_ptr< raw_csr< ValType > > > vcsrb_raw;
  std::vector<compress_t> v_comp_ptr;
  std::vector<ull> block_flops;
  ull gpu_flop_thresh;
  ull max_rcblock_flop;
  
  bool enable_write;

  template<typename T>
  using vec = std::vector<T>;

  static GlobalStorage *Instance();
  static void Init(IdxType blockSizeA, IdxType blockSizeB, IdxType numBlocksA, IdxType numBlocksB, size_t poolSize, int num_workers, csr<SrcEType, ValType> &&a, csc<SrcEType,ValType> &&bT, std::vector<ull> && block_flops, ull gpu_flop_thresh, bool file_write);

private:
  static GlobalStorage* gs;
  GlobalStorage()=delete;
  GlobalStorage(IdxType blockSizeA, IdxType blockSizeB, IdxType numBlocksA, IdxType numBlocksB, size_t poolSize, int num_workers, csr<SrcEType, ValType> &&a, csc<SrcEType,ValType> &&bT, std::vector<ull> && block_flops, ull gpu_flop_thresh, bool file_write);
  ~GlobalStorage();
} ;



template class GlobalStorage<ull, IdxType, float>;
template class GlobalStorage<IdxType, IdxType, float>;

using default_gs_t = GlobalStorage<IdxType, IdxType, float>;
using large_gs_t = GlobalStorage<ull, IdxType, float>;


} // namespace CESpGEMM

