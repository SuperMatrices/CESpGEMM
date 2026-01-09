#include"computeMKL.h"
#include"mklManage.h"
#include"profiler.h"

namespace CESpGEMM
{

// Constructor: Initialize MKL sparse matrix computer
template<typename Gs_t, bool SingleBlock>
MKL_computer<Gs_t, SingleBlock>::MKL_computer(int num_workers, const Gs_t*gs, std::vector<std::unique_ptr<result_t>> &res):
    gs(gs), cscb(gs->csrB_T), csra(gs->csrA), converted(false),
    temp_ptrA(std::unique_ptr<EIdType[]>(new EIdType[gs->blockSizeA+1]))
{
  // Get references to input matrices
  const csc<EIdType,ValType> & cscb_ = *cscb.get();
  const csr<EIdType,ValType> & csra_ = *csra.get();
  // Use 32-bit indices directly, or convert 64-bit to 64-bit format
  if constexpr (sizeof(EIdType) == 4){
    this->cscb_idx_borrow = cscb_.idx;
    this->csra_idx_borrow = csra_.idx;
  }else{
    this->cscb_idx_borrow = gs->cscb_idx_64;
    this->csra_idx_borrow = gs->csra_idx_64;
  }
  // Create MKL CSC matrix for B
  mkl_mat_b = MKL_util::CreateUniqueMatCsc(
    cscb_.nr, cscb_.nc, cscb_.ptr, (EIdType*)this->cscb_idx_borrow, cscb_.val
  ) ;
  mkl_set_num_threads(num_workers);
  for(int i=0;i<2;i++){
    wt[i]= res[i].get();
  }
}

template<typename Gs_t, bool SingleBlock>
// Compute sparse matrix multiplication for a row block using MKL
int MKL_computer<Gs_t, SingleBlock>::compute(int rBid, int buffer) {
  // printf("MKL computer: compute (%d,%d)\n", rBid, buffer);
  // if(!converted){ MKL_util::ConvertCsr(mkl_mat_b); converted = true;}
  auto t0 = std::chrono::steady_clock::now();
  // Get reference to matrix A
  const csr<EIdType, ValType> &csra_ = *csra.get();
  result_t &res=*wt[buffer];
  // Calculate row block boundaries
  IdxType rStart = rBid * gs->blockSizeA, rEnd=std::min(rStart+gs->blockSizeA, csra_.nr), nRows=rEnd-rStart;
  // Get offset into A's data arrays
  EIdType poffset=csra_.ptr[rStart];
  // Build local row pointer for this block
  EIdType* ptrA = temp_ptrA.get();
  for(int i=0;i<=nRows;i++){
    ptrA[i] = csra_.ptr[rStart+i]-poffset;
  }
  // Create MKL CSR matrix for this block of A
  spMat matA = MKL_util::CreateUniqueMatCsr(
    nRows, csra_.nc, ptrA, ((EIdType*)csra_idx_borrow) + poffset, csra_.val + poffset
  );
  
  // mkl_sparse_s_create_csr(&mkl_csr_a, SPARSE_INDEX_BASE_ZERO, nRows, csra_.nc,
  //   reinterpret_cast<MKL_INT*>(ptrA),
  //   reinterpret_cast<MKL_INT*>(ptrA+1),
  //   reinterpret_cast<MKL_INT*>(csra_.idx + poffset),
  //   (csra_.val + poffset)
  // );
  // Perform sparse multiplication: C = A * B
  spMat matC;
  sparse_status_t status = MKL_util::TryMult(matA, mkl_mat_b, matC);
  // sparse_index_base_t indexing;
  // int resultnr,resultnc;
  // int*rowstart,*rowend;
  // int*ref_idx;
  // float*ref_val;
  // mkl_sparse_s_export_csr(mkl_csr_c, &index, &resultnr, &resultnc, &rowstart, &rowend, &ref_idx, &ref_val);
  // Record compute time
  auto t1 = std::chrono::steady_clock::now();
  profiler::Instance().cpu_data.cpu_compute_time += get_chrono_ms(t0, t1);
  // Store result matrix C
  res.setMKLSpMatC1(std::move(matC));
  // Check for MKL errors
  bool has_error = status!=SPARSE_STATUS_SUCCESS;
  // Set block status flags
  res.setBlockStatusAll(128|64|(has_error*32));
  return has_error;
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
