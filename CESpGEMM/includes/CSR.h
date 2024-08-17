#pragma once
#include<cstdint>
#include<memory>
#include<vector>

namespace CESpGEMM{

  
enum class ColRange{
  SMALL,
  BIG,
  TOO
} ;


using IdxType = uint32_t;
using ull = unsigned long long;

template<typename EIdType, typename ValType>
struct coo
{
  void free(){
    if(row){
      delete[]row;
      delete[]col;
      delete[]val;
      row = nullptr;
      col = nullptr;
      val = nullptr;
    }
  }
  IdxType nr, nc;
  EIdType nnz;
  IdxType*row;
  IdxType*col;
  ValType*val;
  coo(size_t nr,size_t nc,size_t nnz):nr(nr),nc(nc),nnz(nnz){
    printf("%lld,%lld,%lld\n", nc, nc, nnz);
    row = new IdxType[nnz];
    col = new IdxType[nnz];
    val = new ValType[nnz];
  }
  coo():nr(0),nc(0),nnz(0),row(nullptr),col(nullptr),val(nullptr){}
  coo(const coo&)=delete;
  coo(coo&& other) noexcept
    : nr(other.nr), nc(other.nc), nnz(other.nnz),
      row(other.row), col(other.col), val(other.val)
  {
    other.row = nullptr;
    other.col = nullptr;
    other.val = nullptr;
  }
  void operator=(const coo&) = delete;
  void operator=(coo&& other) noexcept{
    if(this == &other) return;
    this->free();
    row = other.row; other.row = nullptr; 
    col = other.col; other.col = nullptr;
    val = other.val; other.val = nullptr;
    nr = other.nr;
    nc = other.nc;
    nnz = other.nnz;
  }
  ~coo(){
    free();
  }
} ;

template<typename EIdType, typename ValType>
struct csr
{
  using Etype = EIdType;
  void alloc(IdxType nr,EIdType nnz);
  IdxType nr, nc;
  EIdType nnz;
  EIdType*ptr;
  IdxType*idx;
  ValType*val;
  csr(csr&&) noexcept;
  void operator=(csr &&) noexcept;
  csr(const csr&) = delete;
  void operator=(const csr &) = delete;
  csr():ptr(nullptr),idx(nullptr),val(nullptr){
    // printf("csr() %p\n",this);
  };
  void free(){
    if(ptr){
      delete[]ptr;
      delete[]idx;
      delete[]val;
    }
    ptr=nullptr;
    idx=nullptr;
    val=nullptr;
  }
  ~csr(){
    free();
  }
  csr(IdxType nr, IdxType nc, EIdType nnz);
  void init_from_coo_cpu(const coo<EIdType, ValType>&, bool transposed=false);
  void transpose_self();
  void transpose_self_single_thread();
  std::tuple<EIdType*,IdxType*,ValType*> release(){
    std::tuple<EIdType*,IdxType*,ValType*> ret={ptr,idx,val};
    ptr=nullptr; idx=nullptr; val=nullptr;
    return ret;
  }
  // void operator=(csr&&) noexcept;
} ;

template<typename EIdType, typename ValType>
csr<EIdType, ValType> convert_from_coo_to_vector_csr(const coo<EIdType, ValType> &co, IdxType colBlockSize);

#define csc csr

template<typename TgtEType, typename SrcEType, typename ValType>
csr<TgtEType, ValType> convert_from_csc_to_vector_csr_get_slice(const csc<SrcEType, ValType> &C, IdxType colStart, IdxType colEnd);


template<typename ValType>
struct raw_csr
{
  IdxType *idx;
  ValType *val;
  raw_csr(const raw_csr&)=delete;
  void operator=(const raw_csr&)=delete;
  raw_csr(raw_csr&&)=delete;
  void operator=(raw_csr&&)=delete;
  
  raw_csr():idx(nullptr),val(nullptr){}
  raw_csr(size_t nElems){
    idx = (IdxType*)malloc(sizeof(IdxType) * nElems);
    val = (ValType*)malloc(sizeof(ValType) * nElems);
  }
  static std::unique_ptr<raw_csr> from_pointer(IdxType*indices, ValType* values){
    std::unique_ptr<raw_csr> ret (new raw_csr{});
    ret->idx=indices;
    ret->val=values;
    return ret;
  }
  void refresh_with_size(size_t nElems){
    idx = (IdxType*)realloc(idx, nElems*sizeof(IdxType)) ;
    val = (ValType*)realloc(val, nElems*sizeof(ValType)) ;
  }
  ~raw_csr(){
    if(idx){
      free(idx);
    }
    if(val){
      free(val);
    }
  }
} ;




template struct csr<IdxType, float>;
template struct csr<unsigned long long, float>;

using csr_4f_t = csr<IdxType, float>;
using csr_8f_t = csr<unsigned long long, float>;

}//CESpGEMM