#pragma once
#include<vector>
#include"CSR.h"
#include"mklManage.h"

namespace CESpGEMM{
template<typename EIdType>
struct FlopData
{
  std::vector<EIdType> cbNNZ;
  std::vector<int> cBlockId;
  FlopData(int nb):cbNNZ(nb), cBlockId(nb){

  }
    
} ;




template<typename Gs_t, bool single>
class ComputeResultData
{

  using EIdType = typename Gs_t::eidType;
  using ValType = typename Gs_t::valType;
  using SrcType = typename Gs_t::srcEtype;
  using MKLMat = typename MKLSpMat<SrcType, ValType>::mtype;
private:
  std::vector<std::vector<EIdType>> h_ptrs;
  std::vector<raw_csr<ValType> > h_results;
  std::vector<uint8_t> block_status;
public:
  MKLMat spmatC;
  ComputeResultData(int numBlocksB, int blockSizeA):
    h_ptrs(numBlocksB, std::vector<EIdType>(blockSizeA+1)),
    h_results(numBlocksB),
    block_status(numBlocksB)
  {
  }
  std::vector<EIdType>&getPtr(int idx){
    return h_ptrs.at(idx);
  }
  raw_csr<ValType>&getRawCsr(int idx){
    return h_results.at(idx);
  }
  void setBlockStatusAll(uint8_t val){
    for(uint8_t&s: block_status) s=val;
  }
  void setBlockStatus(int cBid, uint8_t val){
    block_status[cBid] = val;
  }
  uint8_t getBlockStatus(int cBid){
    return block_status[cBid]; 
  }
  void setMKLSpMatC1(MKLMat &&spC){
    spmatC = std::move(spC);
  }
  sparse_matrix_t getSpMatC(){
    return spmatC.get();
  }
} ;


// template<typename Gs_t>
// class ComputeResultData<Gs_t, true>
// {
//   using EIdType = typename Gs_t::eidType;
//   using ValType = typename Gs_t::valType;
//   using spMat_t = typename MKLSpMat<EIdType, ValType>::mtype;
// private:
//   std::vector<EIdType> h_ptrs;
//   raw_csr<ValType> h_results;
//   uint8_t block_status;
//   spMat_t spmatC;
// public:
//   ComputeResultData(int numBlocksB, int blockSizeA):
//     h_ptrs(blockSizeA+1),
//     h_results{},
//     block_status(0)
//   {
//     CHK_ASSERT_EQL(numBlocksB,1);
//   }
//   std::vector<EIdType>&getPtr(int idx){
//     CHK_ASSERT_EQL(idx,0);
//     return h_ptrs;
//   }
//   raw_csr<ValType>&getRawCsr(int idx){
//     CHK_ASSERT_EQL(idx,0);
//     return h_results;
//   }
//   void setBlockStatusAll(uint8_t val){
//     block_status = val;
//   }
//   void setBlockStatus(int cBid, uint8_t val){
//     CHK_ASSERT_EQL(cBid,0);
//     block_status = val;
//   }
//   uint8_t getBlockStatus(int cBid){
//     CHK_ASSERT_EQL(cBid,0);
//     return block_status;
//   }
//   void setMKLSpMatC1(spMat_t &&spC){
//     spmatC = std::move(spC);
//   }
//   sparse_matrix_t getSpMatC(){
//     return spmatC.get();
//   }
  
// } ;




template<typename EIdType, typename ValType>
struct MergedResultData
{
  EIdType* block_nnz;
  
  std::unique_ptr< raw_csr<ValType> > io_buffer;
  size_t capacity;
  MergedResultData(size_t reserved_buffer_size, EIdType*blk_nnz) : io_buffer( std::make_unique<raw_csr<ValType>>(reserved_buffer_size)), block_nnz(blk_nnz){
    capacity = reserved_buffer_size;
  }
  void resize(size_t sz){
    if(capacity<sz){
      printf("merged result data: resize to size: %lld\n", sz);
      capacity = sz;
      std::unique_ptr< raw_csr<ValType> > new_buffer = std::make_unique<raw_csr<ValType>>(sz) ;
      io_buffer.swap(
        new_buffer
      ) ;
    }
  }
} ;

struct EmptyData
{
     
} ;

}//namespace
