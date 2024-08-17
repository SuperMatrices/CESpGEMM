#pragma once
#include"Storage.h"
#include"WritableStorage.h"
#include<vector>
#include"profiler.h"
#include"FileIO.h"

namespace CESpGEMM
{

template<class Gs_t, bool single>
struct MergeWorker
{
  using MergeIdx = typename Gs_t::srcEtype;
	using EIdType = typename Gs_t::eidType;
  using ValType = typename Gs_t::valType;
	using Rd_t = ComputeResultData<Gs_t, single>;
	using Wt_t = MergedResultData<MergeIdx, ValType>;
	Gs_t *gs;
	std::vector<Rd_t*>rd;
	std::vector<Wt_t*>wt;
  
  using fwriter_t = IO::FileWriter<MergeIdx, ValType>;

	MergeIdx *ptr;
	MergeIdx blocknnz_prefsum;

	std::unique_ptr<fwriter_t> p_writer;

  using mkl_func=MKL_Func_Factory<sizeof(MergeIdx)==8>;
  using mkl_int_type = std::conditional_t< sizeof(MergeIdx)==4 , MKL_INT, MKL_INT64 >;

	MergeWorker(Gs_t *gs, std::vector<std::unique_ptr<Rd_t>>&from_data, std::vector<Wt_t>&to_data, const std::string &name):
		gs(gs),rd(from_data.size()), wt(to_data.size())
	{
    if(gs->enable_write){
      printf("creating pwriter...\n");
      p_writer = std::make_unique<fwriter_t>(name);
      printf("pwriter.get=%p\n", p_writer.get());
    }

		blocknnz_prefsum=0;
		for(int i=0;i<from_data.size();i++){
			rd[i]=from_data[i].get();
		}
		for(int i=0;i<to_data.size();i++){
			wt[i]=&to_data[i];
		}	
		ptr = new MergeIdx[gs->blockSizeA+1];
	}

  MergeWorker(MergeWorker&&rhs){
    gs = rhs.gs; rhs.gs = nullptr;
    ptr = rhs.ptr; rhs.ptr = nullptr;
    blocknnz_prefsum = rhs.blocknnz_prefsum;
    p_writer = std::move(rhs.p_writer);
    rd = std::move(rhs.rd);
    wt = std::move(rhs.wt);
  }


	
	void work(int tid, int buffer){
    printf("merge work at (%d,%d)\n", tid, buffer);
    // printf("pwriter=%p!\n", p_writer.get());
		int rBid = tid;
    using std::vector;
    Rd_t *buffer_rd = rd[buffer];

    // std::vector<IdxType>current(nbB);
    // std::vector<IdxType>nxt;
    // for(IdxType cBid=0;cBid<nbB;cBid++){
    //   current[cBid] = cBid;
    // }

		IdxType L = rBid * gs->blockSizeA;
    IdxType R = std::min(L+gs->blockSizeA, gs->csrA.nr), block_nr = R-L;
    Rd_t & rd_src = *rd[buffer];
    Wt_t & w_tgt = *wt[buffer];
    for(IdxType r=0;r<=block_nr;r++){
      ptr[r] = 0;
    }
    for(IdxType cBid=0;cBid<gs->numBlocksB;cBid++){
      while(! (rd_src.getBlockStatus(cBid) & 64) ) std::this_thread::yield();
    }

    if(rd_src.getBlockStatus(0) & 128){//merge cpu results 
      auto t0 = std::chrono::steady_clock::now();
      sparse_matrix_t spmatC_p = rd_src.getSpMatC();
      mkl_int_type * c_ptr, *c_ptr_end;
      mkl_int_type * c_idx;
      float *c_val;
      sparse_index_base_t indexing;
      mkl_int_type  nr, nc;
      mkl_func::export_csr(spmatC_p, &indexing, &nr, &nc, &c_ptr, &c_ptr_end, &c_idx, &c_val);
      for(IdxType r=0;r<block_nr;r++){
        this->ptr[r] = c_ptr[r] + blocknnz_prefsum;
      }
      this->ptr[block_nr] = c_ptr_end[block_nr-1] + blocknnz_prefsum;
      w_tgt.block_nnz[rBid] = c_ptr_end[block_nr-1];
      blocknnz_prefsum = this->ptr[block_nr];
      auto t1 = std::chrono::steady_clock::now();
      fwriter_t &fw = *p_writer;
      if(gs->enable_write){
        fwriter_t &fw = *p_writer.get();
        fw.write_block_csr(block_nr, ptr, reinterpret_cast<IdxType*>(c_idx), c_val);
        if(rBid == gs->numBlocksA-1){
          fw.write_single_ptrval(&blocknnz_prefsum);
        }
      }
      auto t2 = std::chrono::steady_clock::now();
      profiler::Instance().merge_time += get_chrono_ms(t0, t1);
      profiler::Instance().io_time += get_chrono_ms(t1,t2);
      return ;
    }

    auto t0 = std::chrono::steady_clock::now();
    {// compute GPU blocks's merged ptr
      for(IdxType cBid=0;cBid<gs->numBlocksB;cBid++){
        const std::vector<EIdType> &ptrCblock = rd_src.getPtr(cBid);
        for(IdxType r=0;r<block_nr;r++) this->ptr[r+1] += ptrCblock[r+1]-ptrCblock[r];
      }
      for(IdxType r=0;r<block_nr;r++) this->ptr[r+1] += ptr[r];
    }
    //merge gpu results

    printf("rid %d: ptrnr=%d\n",rBid, ptr[block_nr]);

    // wt[buffer]->resize(ptr[block_nr]);
    w_tgt.resize(ptr[block_nr]);
    // printf("w_tgt.cap:%lld\n", w_tgt.capacity);
    raw_csr<ValType> &csr_handle = *w_tgt.io_buffer.get();
    w_tgt.block_nnz[rBid] = ptr[block_nr];

    for(IdxType r=0;r<block_nr;r++){
      MergeIdx offset=ptr[r];
      for(IdxType cb=0;cb<gs->numBlocksB;cb++){
        const std::vector<EIdType> &ptrCblock = rd_src.getPtr(cb);
        const raw_csr<ValType> & csrBlockC = rd_src.getRawCsr(cb);
        IdxType *idx = csrBlockC.idx + ptrCblock[r];
        ValType *val = csrBlockC.val + ptrCblock[r];
        IdxType sz = ptrCblock[r+1] - ptrCblock[r];

        // printf("sz=%d, stat=%d\n", sz, status);
        for(IdxType j=0;j<sz;j++){
          CHK_ASSERT_LESS((ull)(offset+j), (ull)w_tgt.capacity);
          csr_handle.idx[offset + j] = idx[j] + cb * gs->blockSizeB;
          csr_handle.val[offset + j] = val[j];
        }
        offset += sz;
      }
    }
    for(int i=0;i<block_nr;i++){
      ptr[i] += blocknnz_prefsum;
    }
    ptr[block_nr] += blocknnz_prefsum;
    blocknnz_prefsum = ptr[block_nr];
    auto t1 = std::chrono::steady_clock::now();

    if(gs->enable_write){
      fwriter_t &fw = *p_writer.get();
      fw.write_block_csr(block_nr, ptr, csr_handle.idx, csr_handle.val);
      if(rBid == gs->numBlocksA-1){
        fw.write_single_ptrval(&blocknnz_prefsum);
      }
    }

    auto t2 = std::chrono::steady_clock::now();
    profiler::Instance().merge_time += get_chrono_ms(t0, t1);
    profiler::Instance().io_time += get_chrono_ms(t1, t2);
		// printf("merger finished (%d,%d)\n", tid, buffer);
	}
	void finish(){

	}
	~MergeWorker(){
		delete[]ptr;
	}
} ;

template<class Gs_t>
struct MergeWorker<Gs_t, true>
{
  using MergeIdx = typename Gs_t::srcEtype;
  using EIdType = typename Gs_t::eidType;
  using ValType = typename Gs_t::valType;
  using Rd_t = ComputeResultData<Gs_t, true>;
  using Wt_t = MergedResultData<MergeIdx, ValType>;
  Gs_t *gs;
  std::vector<Rd_t*>rd;
  std::vector<Wt_t*>wt;
  using fwriter_t = IO::FileWriter<MergeIdx, ValType>;
  MergeIdx *ptr;
  MergeIdx blocknnz_prefsum;
  std::unique_ptr<fwriter_t>p_writer;
  using mkl_func=MKL_Func_Factory<sizeof(MergeIdx)==8>;
  using mkl_int_type = std::conditional_t< sizeof(MergeIdx)==4 , MKL_INT, MKL_INT64 >;

  MergeWorker(Gs_t *gs, std::vector<std::unique_ptr<Rd_t>>&from_data, std::vector<Wt_t>&to_data, const std::string &name):
		gs(gs),rd(from_data.size()),wt(to_data.size())
	{
    if(gs->enable_write){
      printf("creating pwriter...\n");
      p_writer = std::make_unique<fwriter_t>(name);
      printf("pwriter.get=%p\n", p_writer.get());
    }

		blocknnz_prefsum=0;
		for(int i=0;i<from_data.size();i++){
			rd[i]=from_data[i].get();
      wt[i]=&to_data[i];
		}
		ptr = new MergeIdx[gs->blockSizeA+1];
	}
  MergeWorker(MergeWorker&&rhs)
  {
    gs = rhs.gs; rhs.gs = nullptr;
    ptr = rhs.ptr; rhs.ptr = nullptr;
    blocknnz_prefsum = rhs.blocknnz_prefsum;
    p_writer = std::move(rhs.p_writer);
    rd = std::move(rhs.rd);
    wt = std::move(rhs.wt);
  }


  void work(int tid, int buffer){
    printf("merge(singleBlock ver) worker at (%d, %d)\n", tid, buffer);
    int rBid = tid;
    using std::vector;
    Rd_t& buffer_rd = *rd[buffer];
    Wt_t& w_tgt= *wt[buffer];
    printf("merging results from rBid %d\n", rBid);
    IdxType L = rBid * gs->blockSizeA, R = std::min(L+gs->blockSizeA, gs->csrA.nr), block_nr = R-L;
    while(!(buffer_rd.getBlockStatus(0) & 64)) std::this_thread::yield();
    uint8_t block_status = buffer_rd.getBlockStatus(0);
    if( block_status == 255){
      return;
    }
    if(block_status & 128){
      auto t0 = std::chrono::steady_clock::now();
      sparse_matrix_t spmatC_p = buffer_rd.getSpMatC();
      mkl_int_type* c_ptr, *c_ptr_end;
      mkl_int_type* c_idx;
      float *c_val;
      sparse_index_base_t indexing;
      mkl_int_type nr, nc;
      mkl_func::export_csr(spmatC_p, &indexing, &nr, &nc, &c_ptr, &c_ptr_end, &c_idx, &c_val);
      for(IdxType r=0;r<block_nr;r++){
        this->ptr[r] = c_ptr[r] + blocknnz_prefsum;
      }
      
      this->ptr[block_nr] = c_ptr_end[block_nr-1] + blocknnz_prefsum;
      blocknnz_prefsum = this->ptr[block_nr];
      w_tgt.block_nnz[rBid] = c_ptr_end[block_nr-1];

      auto t1 = std::chrono::steady_clock::now();
      fwriter_t &fw = *p_writer;
      if(gs->enable_write){
        fwriter_t &fw = *p_writer.get();
        fw.write_block_csr(block_nr, ptr, reinterpret_cast<IdxType*>(c_idx), c_val);
        if(rBid == gs->numBlocksA-1){
          fw.write_single_ptrval(&blocknnz_prefsum);
        }
      }
      auto t2 = std::chrono::steady_clock::now();
      profiler::Instance().merge_time += get_chrono_ms(t0, t1);
      profiler::Instance().io_time += get_chrono_ms(t1, t2);
    }else{
      auto t0 = std::chrono::steady_clock::now();
      std::vector<EIdType>&gpu_ptr = buffer_rd.getPtr(0);
      raw_csr<ValType> &hgpu_csr = buffer_rd.getRawCsr(0);
      
      ptr[0] = 0;
      for(IdxType r=0;r<block_nr;r++){
        ptr[r+1] = (gpu_ptr[r+1] - gpu_ptr[r]) + ptr[r];
      }
      w_tgt.block_nnz[rBid] = ptr[block_nr];
      for(IdxType r=0;r<=block_nr;r++) ptr[r] += blocknnz_prefsum;
      auto t1 = std::chrono::steady_clock::now();
      blocknnz_prefsum = ptr[block_nr];
      if(gs->enable_write){
        fwriter_t &fw = *p_writer.get();
        fw.write_block_csr(block_nr, ptr, hgpu_csr.idx, hgpu_csr.val);
        if(rBid == gs->numBlocksA -1){
          fw.write_single_ptrval(&blocknnz_prefsum);
        }
      }
      auto t2 = std::chrono::steady_clock::now();
      profiler::Instance().merge_time += get_chrono_ms(t0, t1);
      profiler::Instance().io_time += get_chrono_ms(t1, t2);
    }
    // printf("merger finished (%d, %d)\n", tid, buffer);       
  }
  void finish(){

  }
  ~MergeWorker(){
    delete[]ptr;
  }

} ;



}
