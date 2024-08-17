#include"computeMKL.h"
#include"mklManage.h"
#include"profiler.h"

namespace CESpGEMM
{

template<typename Gs_t, bool SingleBlock>
MKL_computer<Gs_t, SingleBlock>::MKL_computer(int num_workers, const Gs_t*gs, std::vector<std::unique_ptr<result_t>> &res):
    gs(gs), cscb(gs->csrB_T), csra(gs->csrA), converted(false),
    temp_ptrA(std::unique_ptr<EIdType[]>(new EIdType[gs->blockSizeA+1]))
{
  if constexpr (sizeof(EIdType) == 4){
    this->cscb_idx_borrow = cscb.idx;
    this->csra_idx_borrow = csra.idx;
  }else{
    this->cscb_idx_borrow = gs->cscb_idx_64;
    this->csra_idx_borrow = gs->csra_idx_64;
  }
  mkl_mat_b = MKL_util::CreateUniqueMatCsc(
    cscb.nr, cscb.nc, cscb.ptr, (EIdType*)this->cscb_idx_borrow, cscb.val
  ) ;
  mkl_set_num_threads(num_workers);
  for(int i=0;i<2;i++){
    wt[i]= res[i].get();
  }
}

template<typename Gs_t, bool SingleBlock>
void MKL_computer<Gs_t, SingleBlock>::compute(int rBid, int buffer) {
  printf("MKL computer: compute (%d,%d)\n", rBid, buffer);
  // if(!converted){ MKL_util::ConvertCsr(mkl_mat_b); converted = true;}
  auto t0 = std::chrono::steady_clock::now();
  result_t &res=*wt[buffer];
  IdxType rStart = rBid * gs->blockSizeA, rEnd=std::min(rStart+gs->blockSizeA, csra.nr), nRows=rEnd-rStart;
  EIdType poffset=csra.ptr[rStart];
  EIdType* ptrA = temp_ptrA.get();
  for(int i=0;i<=nRows;i++){
    ptrA[i] = csra.ptr[rStart+i]-poffset;
  }
  spMat matA = MKL_util::CreateUniqueMatCsr(
    nRows, csra.nc, ptrA, ((EIdType*)csra_idx_borrow) + poffset, csra.val + poffset
  );
  
  // mkl_sparse_s_create_csr(&mkl_csr_a, SPARSE_INDEX_BASE_ZERO, nRows, csra.nc,
  //   reinterpret_cast<MKL_INT*>(ptrA),
  //   reinterpret_cast<MKL_INT*>(ptrA+1),
  //   reinterpret_cast<MKL_INT*>(csra.idx + poffset),
  //   (csra.val + poffset)
  // );
  spMat matC = MKL_util::Mult(matA, mkl_mat_b);
  // sparse_index_base_t indexing;
  // int resultnr,resultnc;
  // int*rowstart,*rowend;
  // int*ref_idx;
  // float*ref_val;
  // mkl_sparse_s_export_csr(mkl_csr_c, &index, &resultnr, &resultnc, &rowstart, &rowend, &ref_idx, &ref_val);
  auto t1 = std::chrono::steady_clock::now();
  profiler::Instance().cpu_compute_time += get_chrono_ms(t0, t1);
  res.setMKLSpMatC1(std::move(matC));
  res.setBlockStatusAll(128|64);
}


template
struct MKL_computer<default_gs_t, false>;
template
struct MKL_computer<default_gs_t, true>;

template
struct MKL_computer<large_gs_t, false>;
template
struct MKL_computer<large_gs_t, true>;

} // namespace CESpGEMM
