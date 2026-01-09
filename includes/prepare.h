#pragma once
#include"Storage.h"
#include"WritableStorage.h"
#include<vector>
#include<iostream>
#include<algorithm>
#include"profiler.h"


namespace CESpGEMM{


template<class Gs_t>
struct PrepareWorker
{
  using SrcEType = typename Gs_t::srcEtype;
  using EIdType = typename Gs_t::eidType;
  using ValType = typename Gs_t::valType;

  using Wt_t = FlopData<EIdType>;
  Gs_t *gs;
  std::vector<Wt_t *>wt;
  
  PrepareWorker(Gs_t *gs, std::vector<Wt_t> & data):gs(gs), wt(data.size()){
    for(int i=0;i<(int)data.size();i++){
      wt[i] = &data[i];
    }
  }

  PrepareWorker(PrepareWorker&& rhs){
    gs = rhs.gs;
    rhs.gs = nullptr;
    wt = std::move(rhs.wt);
  }


  int work(int tid, int buffer){
    const csr<SrcEType,ValType>& csra_ = *gs->csrA.get(); 
    const csc<SrcEType,ValType>& cscb_ = *gs->csrB_T.get(); 

    // printf("prepare work at (%d,%d)\n", tid, buffer);
    int start_row = tid * gs->blockSizeA, end_row = std::min<int>(start_row + gs->blockSizeA, csra_.nr);
    
    using raw_b_t = decltype(gs->vcsrb_raw[0]);
    std::vector<int>& col_indices = wt[buffer]->cBlockId;
    std::vector<EIdType>& cb_nnz = wt[buffer]->cbNNZ;

    IdxType numBlocksB = gs->numBlocksB;
    auto t0=std::chrono::steady_clock::now();
    for(IdxType cbid=0;cbid<numBlocksB;cbid++){
      IdxType cStart = cbid * gs->blockSizeB, cEnd = std::min<IdxType>(cStart+gs->blockSizeB, cscb_.nr);
      cb_nnz[cbid] = cscb_.ptr[cEnd] - cscb_.ptr[cStart];
      col_indices[cbid] = cbid;
    }
    std::sort(col_indices.begin(), col_indices.end(), [& cb_nnz](int i, int j) {
      return cb_nnz[i] > cb_nnz[j]; 
    });
    auto t1=std::chrono::steady_clock::now();
    profiler::Instance().cpu_data.prepare_time += get_chrono_ms(t0,t1);
    return 0;
  }

  void finish(){

  }
};

}//namespace CESpGEMM