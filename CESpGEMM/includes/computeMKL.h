#pragma once
#include"CSR.h"
#include"Storage.h"
#include"WritableStorage.h"
#include"mklManage.h"
#include"profile.h"

namespace CESpGEMM{

template<typename Gs_t, bool SingleBlock>
struct MKL_computer{
using EIdType = typename Gs_t::srcEtype;
using ValType = typename Gs_t::valType;
using MKL_util = MKLSpMat<EIdType,ValType>;
using spMat = typename MKL_util::mtype;
using result_t = ComputeResultData<Gs_t, SingleBlock>;

const Gs_t *gs;
result_t *wt[2];
const csc<EIdType, ValType> &cscb;

const csr<EIdType, ValType> &csra;
void* cscb_idx_borrow;
void* csra_idx_borrow;
// sparse_matrix_t mkl_mat_b;
spMat mkl_mat_b;
bool converted;
std::unique_ptr<EIdType[]> temp_ptrA;


MKL_computer(int num_workers, const Gs_t*gs, std::vector<std::unique_ptr<result_t>> &res);

void compute(int rBid, int buffer);

} ;

}//namespace CESpGEMM