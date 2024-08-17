#include"CSR.h"
#include<omp.h>
#include<cstring>
#include<atomic>
#include"helper.cuh"
#include<limits>

namespace CESpGEMM
{

constexpr int n_omp_threads_util = 8;

template<typename EIdType, typename ValType>
void csr<EIdType, ValType>::alloc(IdxType nr,EIdType nnz){
  ptr = new EIdType[(nr+1)];
  idx = new IdxType[(nnz)];
  val = new ValType[(nnz)];
}


// template<typename EIdType, typename ValType>
// csr<EIdType,ValType>::csr(const csr &c){
//   printf("csr(&) from %p to %p\n", &c, this);
//   nr=c.nr;
//   nc=c.nc;
//   nnz=c.nnz;
  
//   this->alloc(nr, nnz);

//   printf("memcpy time %.3f\n",
//     timeit([&](){
//       memcpy(ptr, c.ptr, sizeof(EIdType) * (nr+1));
//       memcpy(idx, c.idx, sizeof(IdxType) * (nnz));
//       memcpy(val, c.val, sizeof(ValType) * (nnz));
//     })
//   ) ;
// }

template<typename eidType, typename valType>
csr<eidType, valType>::csr(csr && c) noexcept{
  // printf("csr(&&) from %p to %p\n", &c, this);
  nr=c.nr;
  nc=c.nc;
  nnz=c.nnz;
  ptr = c.ptr; c.ptr = nullptr;
  idx = c.idx; c.idx = nullptr;
  val = c.val; c.val = nullptr;
}

template<typename eidType, typename valType>
void csr<eidType, valType>::operator=(csr && c) noexcept{
  printf("operator = csr(&&) from %p to %p\n", &c, this);
  if(this==&c) return;
  this->free();
  nr=c.nr;
  nc=c.nc;
  nnz=c.nnz;
  ptr = c.ptr;  c.ptr = nullptr;
  idx = c.idx;  c.idx = nullptr;
  val = c.val;  c.val = nullptr;
}

template<typename eidType, typename valType>
csr<eidType, valType>::csr(IdxType nr, IdxType nc, eidType nnz):
  nr(nr),nc(nc),nnz(nnz)
{
  printf("csr(%d,%d,%lld) %p\n",nr,nc,nnz, this);
  alloc(nr,nnz);
}

template<typename eidType, typename valType>
void csr<eidType, valType>::init_from_coo_cpu(const coo<eidType, valType> & c, bool transposed){
  nr = c.nr;
  nc = c.nc;
  nnz = c.nnz;
  IdxType* tmp_row = c.row;
  IdxType* tmp_col = c.col;
  printf("init from coo(sizeofEidType:%d): %d,%d,%lld, %p,%p,%p\n", sizeof(eidType), c.nr, c.nc, c.nnz, c.row, c.col, c.val);
  if(transposed){
    std::swap(tmp_row, tmp_col);
    std::swap(nr, nc);
  }
  alloc(nr, nnz);
  std::vector<eidType> stat(nr);

  #pragma omp parallel for num_threads(n_omp_threads_util)
  for(eidType i=0;i<nnz;i++){
    IdxType rid = tmp_row[i];
    IdxType cid = tmp_col[i];
    #pragma omp atomic
    ++stat[rid];
  }
  ptr[0]=0;
  for(IdxType i=0;i<nr;i++){
    ptr[i+1]=ptr[i] + stat[i];
    stat[i]=ptr[i];
  }
  printf("init: nnz=%lld, ptr[nr]=%lld\n", nnz, ptr[nr]);
  CHK_ASSERT_EQL(nnz, ptr[nr]);
  #pragma omp parallel for num_threads(n_omp_threads_util)
  for(eidType i=0;i<nnz;i++){
    eidType s;
    IdxType rid = tmp_row[i];
    #pragma omp critical
      s=stat[rid]++;
    idx[s] = tmp_col[i];
    val[s] = c.val[i];
  }
}

template<typename eidType, typename valType>
void csr<eidType, valType>::transpose_self(){
  // printf("transpose self : nr=%d, nnz=%lld\n", this->nr, this->nnz);
  std::vector<eidType> stat(nc);
  eidType* new_ptr=(new eidType[nc+1]);
  IdxType* new_idx=(new IdxType[nnz]);
  valType* new_val=(new valType[nnz]);

  for(IdxType i=0;i<nc;i++) stat[i]=0;
  #pragma omp parallel for num_threads(n_omp_threads_util)
  for(eidType i=0;i<nnz;i++){
    #pragma omp atomic
    stat[idx[i]]++;
  }

  new_ptr[0]=0;
  for(IdxType i=0;i<nc;i++){
    new_ptr[i+1]=new_ptr[i]+stat[i];
    stat[i]=new_ptr[i];
  }
  #pragma omp parallel for num_threads(n_omp_threads_util)
  for(IdxType i=0;i<nr;i++){
    eidType s;
    for(eidType j=ptr[i];j<ptr[i+1];j++){
      #pragma omp critical
      {
        s=stat[idx[j]]++;
      }
      new_idx[s] = i;
      new_val[s] = val[j];
    }
  }
  std::swap(nr, nc);
  std::swap(ptr, new_ptr);
  std::swap(idx, new_idx);
  std::swap(val, new_val);
  delete[] new_ptr;
  delete[] new_idx;
  delete[] new_val;
  // printf("trans: nnz=%lld, ptr[nr]=%lld\n", nnz, ptr[nr]);
}

template<typename eidType, typename valType>
void csr<eidType, valType>::transpose_self_single_thread(){
  // printf("transpose self : nr=%d, nnz=%lld\n", this->nr, this->nnz);
  std::vector<eidType> stat(nc);
  eidType* new_ptr=(new eidType[nc+1]);
  IdxType* new_idx=(new IdxType[nnz]);
  valType* new_val=(new valType[nnz]);

  for(IdxType i=0;i<nc;i++) stat[i]=0;
  for(eidType i=0;i<nnz;i++) stat[idx[i]]++;

  new_ptr[0]=0;
  for(IdxType i=0;i<nc;i++){
    new_ptr[i+1]=new_ptr[i]+stat[i];
    stat[i]=new_ptr[i];
  }

  for(IdxType i=0;i<nr;i++){
    eidType s;
    for(eidType j=ptr[i];j<ptr[i+1];j++){
      s=stat[idx[j]]++;
      new_idx[s] = i;
      new_val[s] = val[j];
    }
  }
  std::swap(nr, nc);
  std::swap(ptr, new_ptr);
  std::swap(idx, new_idx);
  std::swap(val, new_val);
  delete[] new_ptr;
  delete[] new_idx;
  delete[] new_val;
  // printf("trans: nnz=%lld, ptr[nr]=%lld\n", nnz, ptr[nr]);
}

template<typename EIdType, typename ValType>
std::vector<csr<EIdType, ValType>> convert_from_coo_to_vector_csr(const coo<EIdType, ValType> &co, IdxType colBlockSize) [[deprecated("not used")]]
{
  IdxType nr = co.nr;
  IdxType nc = co.nc, nb = (nc+colBlockSize-1)/colBlockSize;
  std::vector<csr<EIdType, ValType>>vcsr(nb);
  std::vector<std::vector<IdxType>> stat(nb, std::vector<IdxType>(nr));
  for(IdxType i=0,L=0;i<nb;i++,L+=colBlockSize){
    IdxType R = std::min(L+colBlockSize, nc);
    vcsr[i].nr = nr;
    vcsr[i].nc = R-L;
    vcsr[i].ptr = new EIdType[nr+1];
  }

  // #pragma omp parallel for num_threads(n_omp_threads_util)
  for(EIdType i=0;i<co.nnz;i++){
    IdxType r = co.row[i];
    IdxType c = co.col[i];
    // #pragma omp atomic
    stat[c/colBlockSize][r] ++;
  }
  
  // #pragma omp parallel for num_threads(n_omp_threads_util)
  for(EIdType cb=0;cb<nb;cb++){
    auto &t=vcsr[cb];
    auto &v=stat[cb];
    t.ptr[0]=0;
    for(IdxType i=0;i<nr;i++){
      t.ptr[i+1] = t.ptr[i] + v[i];
      v[i] = t.ptr[i];
    }
    t.nnz = t.ptr[nr];
    t.idx = new IdxType[t.nnz];
    t.val = new ValType[t.nnz];
  }

  // #pragma omp parallel for num_threads(n_omp_threads_util)
  for(EIdType i=0;i<co.nnz;i++){
    IdxType r = co.row[i];
    IdxType c = co.col[i];
    EIdType s;
    // #pragma omp critical
    // {
      s = stat[c/colBlockSize][r] ++;
    // }
    vcsr[c/colBlockSize].idx[s] = c;
    vcsr[c/colBlockSize].val[s] = co.val[i];
  }
  return std::move(vcsr);
}

template<typename TgtEType, typename SrcEType, typename ValType>
csr<TgtEType, ValType> convert_from_csc_to_vector_csr_get_slice(const csc<SrcEType, ValType> &C, IdxType colStart, IdxType colEnd){
  static_assert(sizeof(TgtEType) <= sizeof(SrcEType));
  csr<TgtEType, ValType> ret;
  const SrcEType pStart = C.ptr[colStart], pEnd = C.ptr[colEnd];
  printf("colstart%d colend%d nr%d nc%d nnz%lld\n", colStart, colEnd, colEnd-colStart, C.nc, pEnd-pStart);
  CHK_ASSERT_LESS((SrcEType)(pEnd-pStart), (SrcEType) std::numeric_limits<TgtEType>::max());
  ret.alloc(colEnd-colStart, pEnd - pStart);
  ret.nr = colEnd-colStart;
  ret.nc = C.nc;
  ret.nnz = pEnd-pStart;

  for(IdxType i=colStart;i<=colEnd;i++){
    ret.ptr[i-colStart] = C.ptr[i] - pStart;
  }
  for(SrcEType i=pStart;i<pEnd;i++) ret.idx[i-pStart] = C.idx[i];
  for(SrcEType i=pStart;i<pEnd;i++) ret.val[i-pStart] = C.val[i];
  ret.transpose_self_single_thread();
  return std::move(ret);
}



template
csr<unsigned int, float> convert_from_csc_to_vector_csr_get_slice(const csc<unsigned int, float> &C, IdxType colStart, IdxType colEnd);
template
csr<unsigned int, float> convert_from_csc_to_vector_csr_get_slice(const csc<unsigned long long, float> &C, IdxType colStart, IdxType colEnd);


} // namespace CESpGEMM
