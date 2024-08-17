#include"Storage.h"
#include"CSR.h"
#include"generator.h"
#include<omp.h>
#include"profiler.h"
#include<cstring>

namespace CESpGEMM
{


template<typename SrcEType, typename EIdType, typename ValType>
GlobalStorage<SrcEType, EIdType, ValType> * GlobalStorage<SrcEType, EIdType, ValType>::gs = nullptr;

template<typename SrcEType, typename EIdType, typename ValType>
void GlobalStorage<SrcEType, EIdType, ValType>::Init(IdxType blockSizeA, IdxType blockSizeB, IdxType numBlocksA, IdxType numBlocksB, size_t poolSize, int num_workers, csr<SrcEType, ValType> &&a, csc<SrcEType,ValType> &&bT, std::vector<ull> && block_flops, ull gpu_flop_thresh, bool file_write){
  GlobalStorage<SrcEType, EIdType, ValType>::gs = new GlobalStorage(blockSizeA, blockSizeB, numBlocksA, numBlocksB, poolSize, num_workers, std::move(a), std::move(bT), std::move(block_flops), gpu_flop_thresh, file_write);
}

template<typename SrcEType, typename EIdType, typename ValType>
GlobalStorage<SrcEType, EIdType, ValType> * GlobalStorage<SrcEType, EIdType, ValType>::Instance(){
  return GlobalStorage<SrcEType, EIdType, ValType>::gs;
}

template<typename SrcEType, typename EIdType, typename ValType>
GlobalStorage<SrcEType, EIdType, ValType>::~GlobalStorage(){
  printf("~GlobalStorage\n");
}

template<typename SrcEType, typename EIdType, typename ValType>
GlobalStorage<SrcEType, EIdType, ValType>::GlobalStorage(IdxType blockSizeA, IdxType blockSizeB, IdxType numBlocksA, IdxType numBlocksB, size_t poolSize, int num_workers, csr<SrcEType, ValType> &&a, csc<SrcEType,ValType> &&bT, std::vector<ull> && block_flops, ull gpu_flop_thresh, bool file_write)
: 
  // io_collector_buffer(poolSize),
  pool_size(poolSize),
  csrA(std::move(a)),
  vcsrb_raw(numBlocksB),
  csrB_T(std::move(bT)),
  block_flops(std::move(block_flops)),
  numBlocksA(numBlocksA),
  numBlocksB(numBlocksB),
  blockSizeA(blockSizeA),
  blockSizeB(blockSizeB),
  num_workers(num_workers),
  v_comp_ptr(numBlocksB),
  enable_write(file_write),
  gpu_flop_thresh(gpu_flop_thresh),
  max_rcblock_flop(0)
{
  constexpr int n_omp_threads_util = 8;
  profiler &prf=profiler::Instance();
  printf("! nw*bs=%lld, nb=%d\n", 1ll*num_workers*blockSizeB, numBlocksB);
  ull total_bytes_after_comp_all = 0;
  std::vector< std::vector<double> > omp_worker_compress_time(n_omp_threads_util);
  if constexpr (sizeof(SrcEType) == 8){
    csra_idx_64 = new ull[csrA.nnz];
    cscb_idx_64 = new ull[csrB_T.nnz];
    for(ull i=0;i<csrB_T.nnz;i++){
      cscb_idx_64[i] = csrB_T.idx[i];
    }
    for(ull i=0;i<csrA.nnz;i++){
      csra_idx_64[i] = csrA.idx[i];
    }
    
    // std::memcpy(cscb_idx_64, csrB_T.idx, sizeof(ull)*csrB_T.nnz);
    // std::memcpy(csra_idx_64, csrA.idx, sizeof(ull)*csrA.nnz);
  }
  
  if(1 || blockSizeB * 1ll * numBlocksB < 60'000'000 || numBlocksB <= 1000){
    IdxType nrB = csrB_T.nc;
    IdxType*aux = new IdxType[ (nrB+1) * 1ll * (n_omp_threads_util) ];
    std::atomic_int shared_idx=0;
    ull&max_flop = max_rcblock_flop;


    auto t0 = std::chrono::steady_clock::now();
    #pragma omp parallel num_threads(n_omp_threads_util) shared(shared_idx)
    {
      int tid = omp_get_thread_num();
      std::vector<ull> block_flops(numBlocksA,0ull);
      ull local_flops = 0;
      ull total_bytes_after_comp = 0;
      for(;;){
        int i = shared_idx.fetch_add(1);
        #ifdef TEST_PREPROC
        // printf("barrier! now %d, mine%d\n", shared_idx.load(), i);
        #pragma omp barrier
        // printf("barrierend\n");
        // #pragma omp barrier
        #endif
        bool should_break = shared_idx.load()>=numBlocksB;
        if(i<numBlocksB){
          csr<EIdType, ValType> p_csrB = convert_from_csc_to_vector_csr_get_slice<EIdType, SrcEType, ValType>(csrB_T, i * blockSizeB, std::min((i+1)*blockSizeB, csrB_T.nr) );
          
          for(IdxType rid=0;rid<csrA.nr;rid++){
            ull flops =0 ;
            for(SrcEType o=csrA.ptr[rid], oend=csrA.ptr[rid+1];o<oend; o++){
              IdxType cid = csrA.idx[o];
              flops += p_csrB.ptr[cid+1]-p_csrB.ptr[cid];
            }
            block_flops[rid/blockSizeA] += flops;
            // printf("rid=%d, flop=%lld\n", rid, flops);
          }
          for(IdxType rb=0;rb<numBlocksA;rb++){
            // printf("block_flop%d=%lld\n", rb, block_flops[rb]);
            local_flops = std::max(local_flops, block_flops[rb]);
            block_flops[rb] = 0;
          }

          IdxType nr = p_csrB.nr;
          
          auto[ptr, idx, val] = p_csrB.release();
          // printf("omp worker : %d, nnz=%lld, local_flops=%lld\n", omp_get_thread_num(), ptr[nr], local_flops);
          vcsrb_raw[i] = std::move(raw_csr<ValType>::from_pointer(idx, val));
          compress_t& comp_i = v_comp_ptr.at(i);

          auto t0 = std::chrono::steady_clock::now();
          compress_ptr<EIdType, ValType, compress_t>(nr, ptr, comp_i, aux + tid*(nrB+1));
          auto t1 = std::chrono::steady_clock::now();
          ull bytes_after_comp = (comp_i.num_segs+1ll) * 3* sizeof(int) + (comp_i.bytes_of_data + comp_i.nbytes_control) ;
          total_bytes_after_comp += bytes_after_comp;
          omp_worker_compress_time[tid].emplace_back(get_chrono_ms(t0,t1));
          delete[]ptr;
        }
        if(should_break) break;
      }
      // printf("omp worker : %d, flops=%lld\n", omp_get_thread_num(), local_flops);
      #pragma omp critical
      {
        max_flop = max_flop > local_flops ? max_flop : local_flops;
      }
      
      #pragma omp atomic
      
      total_bytes_after_comp_all += total_bytes_after_comp;
    
    }
    auto t1 = std::chrono::steady_clock::now();
    prf.convert_vcsr_time += get_chrono_ms(t0, t1);
    printf("max flops = %lld\n", this->max_rcblock_flop);
    delete[]aux;
  }
  else{
    printf(" NOT IMPLENTED!");
    throw;
  }
  ull raw_ptr_bytes = numBlocksB * 1ll * sizeof(EIdType) * csrB_T.nc;
  profiler::Instance().compress_ratio = raw_ptr_bytes * 1.0 / total_bytes_after_comp_all;
  profiler::Instance().compressd_len = total_bytes_after_comp_all;
  printf("csrA.ptr%p numBlocksA%d\n", csrA.ptr, numBlocksA);
  printf("compress time %.4f,convert_vcsr_time %.4f\n", prf.compress_time, prf.convert_vcsr_time);
  printf("%p %d\n", &csrA, csrA.nnz);
  double total_millisecs = 0;
  int max_len=0;
  for(int i=0;i<n_omp_threads_util;i++) max_len = std::max(max_len, (int)omp_worker_compress_time.size());
  for(int i=0;i<max_len;i++){
    double tmax=0;
    for(int t=0;t<n_omp_threads_util;t++){
      if(omp_worker_compress_time[t].size() > i){
        tmax = std::max(tmax, omp_worker_compress_time[t][i]);
      }
    }
    total_millisecs += tmax;
  }
  profiler::Instance().throughput_kBpS = raw_ptr_bytes / total_millisecs;
}

}//namespace CESpGEMM