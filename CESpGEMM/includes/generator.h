#pragma once
#include"compressor.h"
#include"compress4B.h"
#include"Storage.h"
#include"CSR.h"

namespace CESpGEMM
{
template<typename Gs_t=default_gs_t>
struct GenWorker
{
  using EIdType = typename Gs_t::eidType;
  using ValType = typename Gs_t::valType;
  using csr_t = csr<EIdType, ValType>;
  Gs_t *gs;
  GenWorker(Gs_t*gs):gs(gs){}
  GenWorker(GenWorker && rhs){
    gs = rhs.gs;
  }
  void work(int rid, int buffer){
    printf("generator at (%d, %d)\n", rid, buffer);
  }

} ;


template<typename EIdType, typename ValType>
bool validate_decompress_on_gpu(const csr<EIdType, ValType> &c, const compress_t & cmp);


template<typename EIdType, typename ValType, typename Compressor>
inline void compress_ptr(IdxType nr, EIdType* cPtr, Compressor &cmp, IdxType* aux){
  int compLen =nr + 1;
  aux[0]=0;
  for(int i=1;i<compLen;i++){
    aux[i] = cPtr[i] - cPtr[i-1];
  }
  cmp.compress_with_zero_per_seg(compLen, aux, 16, 2);
}

} // namespace CESpGEMM
// coo->csr used:854828.099 /mnt/sda2/share/sparsemat/suitesparse_big/mycielskian20/mycielskian20.mtx 100000