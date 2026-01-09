#include"Storage.h"
#include"CSR.h"
#include"generator.h"
#include<omp.h>
#include"profiler.h"
#include<cstring>

namespace CESpGEMM
{


// GlobalStorage: Singleton class managing all matrix data and computation parameters
template<typename SrcEType, typename EIdType, typename ValType>
GlobalStorage<SrcEType, EIdType, ValType> * GlobalStorage<SrcEType, EIdType, ValType>::gs = nullptr;

// Initialize global storage with matrix blocks and compression parameters
template<typename SrcEType, typename EIdType, typename ValType>
void GlobalStorage<SrcEType, EIdType, ValType>::Init(IdxType blockSizeA, IdxType blockSizeB, IdxType numBlocksA, IdxType numBlocksB, size_t poolSize, int num_workers, shared_ptr<csr<SrcEType, ValType>> a, shared_ptr<csc<SrcEType,ValType>> bT, std::vector<ull> && block_flops, ull gpu_flop_thresh, size_t max_gpu_bytes, bool file_write, CompInfo comp_info){
  if(GlobalStorage<SrcEType, EIdType, ValType>::gs != nullptr){
    delete GlobalStorage<SrcEType, EIdType, ValType>::gs;
  }
  GlobalStorage<SrcEType, EIdType, ValType>::gs = new GlobalStorage(blockSizeA, blockSizeB, numBlocksA, numBlocksB, poolSize, num_workers, std::move(a), std::move(bT), std::move(block_flops), gpu_flop_thresh, max_gpu_bytes, file_write, comp_info);
}

// Get singleton instance of GlobalStorage
template<typename SrcEType, typename EIdType, typename ValType>
GlobalStorage<SrcEType, EIdType, ValType> * GlobalStorage<SrcEType, EIdType, ValType>::Instance(){
  return GlobalStorage<SrcEType, EIdType, ValType>::gs;
}

template<typename SrcEType, typename EIdType, typename ValType>
GlobalStorage<SrcEType, EIdType, ValType>::~GlobalStorage(){
  if constexpr (sizeof(SrcEType) == 8){
    if(csra_idx_64) delete[]csra_idx_64;
    if(cscb_idx_64) delete[]cscb_idx_64;
  }
}

// Constructor: Initialize storage, compress matrix B, compute flops per block
template<typename SrcEType, typename EIdType, typename ValType>
GlobalStorage<SrcEType, EIdType, ValType>::GlobalStorage(IdxType blockSizeA, IdxType blockSizeB, IdxType numBlocksA, IdxType numBlocksB, size_t poolSize, int num_workers, shared_ptr<csr<SrcEType, ValType>> a, shared_ptr<csc<SrcEType,ValType>> bT, std::vector<ull> && block_flops, ull gpu_flop_thresh, size_t max_gpu_bytes, bool file_write, CompInfo comp_info)
: 
  // io_collector_buffer(poolSize),
  pool_size(poolSize),
  // cpu_tile_idx(numBlocksB, std::vector<CVecType<IdxType>>(blockSizeA,CVecType<IdxType>())),
  // cpu_tile_val(numBlocksB, std::vector<CVecType<ValType>>(blockSizeA,CVecType<ValType>())),
  // gpu_tile_idx(numBlocksB, std::vector<CVecType<IdxType>>(blockSizeA,CVecType<IdxType>())),
  // gpu_tile_val(numBlocksB, std::vector<CVecType<ValType>>(blockSizeA,CVecType<ValType>())),
  csrA(a),
  vcsrb_raw(numBlocksB),
  csrB_T(bT),
  block_flops(std::move(block_flops)),
  numBlocksA(numBlocksA),
  numBlocksB(numBlocksB),
  blockSizeA(blockSizeA),
  blockSizeB(blockSizeB),
  num_workers(num_workers),
  cmpInfo(comp_info),
  v_comp_ptr(numBlocksB, compress_t(comp_info.seglen)),
  enable_write(file_write),
  gpu_flop_thresh(gpu_flop_thresh),
  max_rcblock_flop(0)
{
  constexpr int n_omp_threads_util = 8;
  profiler &prf=profiler::Instance();
  ull total_bytes_after_comp_all = 0;
  std::vector< std::vector<double> > omp_worker_compress_time(n_omp_threads_util);
  auto & csra_ = *csrA.get();
  auto & csrb_t_ = *csrB_T.get();
  // Convert 64-bit indices to 64-bit format for GPU compatibility
  if constexpr (sizeof(SrcEType) == 8){
    csra_idx_64 = new ull[csra_.nnz];
    cscb_idx_64 = new ull[csrb_t_.nnz];
    for(ull i=0;i<csrb_t_.nnz;i++){
      cscb_idx_64[i] = csrb_t_.idx[i];
    }
    for(ull i=0;i<csra_.nnz;i++){
      csra_idx_64[i] = csra_.idx[i];
    }
    
    // std::memcpy(cscb_idx_64, csrb_t_.idx, sizeof(ull)*csrb_t_.nnz);
    // std::memcpy(csra_idx_64, csra_.idx, sizeof(ull)*csra_.nnz);
  }

  // Calculate maximum flops allowed per GPU block based on available memory
  size_t bytes_for_flop = max_gpu_bytes - sizeof(ValType) * (nGridsMerge * bT.get()->nr);
  bytes_for_flop /= 3;
  this->max_allowed_flop = bytes_for_flop/(sizeof(IdxType) + sizeof(ValType));
  // std::cout<<"max_allowed_flop = "<<max_allowed_flop<<"\n";


  
  // Preprocess matrix B: split into blocks, compress each block, estimate flops
  if(1 || blockSizeB * 1ll * numBlocksB < 60'000'000 || numBlocksB <= 1000){
    IdxType nrB = csrb_t_.nc;
    IdxType*aux = new IdxType[ (nrB+1) * 1ll * (n_omp_threads_util) ];
    std::atomic_int shared_idx=0;
    ull&max_flop = max_rcblock_flop;


    auto t0 = std::chrono::steady_clock::now();
    // Parallel preprocessing of matrix B blocks
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
          // Extract a column block from B and convert to CSR
          csr<EIdType, ValType> p_csrB = convert_from_csc_to_vector_csr_get_slice<EIdType, SrcEType, ValType>(csrb_t_, i * blockSizeB, std::min((i+1)*blockSizeB, csrb_t_.nr) );
          
          // Estimate flops for each row block of A with this B block
          for(IdxType rid=0;rid<csra_.nr;rid++){
            ull flops =0 ;
            for(SrcEType o=csra_.ptr[rid], oend=csra_.ptr[rid+1];o<oend; o++){
              IdxType cid = csra_.idx[o];
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
          
          // Store raw CSR data and compress the ptr array
          auto[ptr, idx, val] = p_csrB.release();
          // printf("omp worker : %d, nnz=%lld, local_flops=%lld\n", omp_get_thread_num(), ptr[nr], local_flops);
          vcsrb_raw[i] = std::move(raw_csr<ValType>::from_pointer(idx, val));
          compress_t& comp_i = v_comp_ptr.at(i);

          // Compress the ptr array using parameterized compression
          auto t0 = std::chrono::steady_clock::now();
          compress_ptr_parameterized<EIdType, ValType, compress_t>(nr, ptr, comp_i, aux + tid*(nrB+1), cmpInfo.min_zero_num, cmpInfo.max_zero_seg);
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
    prf.cpu_data.convert_vcsr_time += get_chrono_ms(t0, t1);
    // printf("max flops = %lld\n", this->max_rcblock_flop);
    delete[]aux;
  }
  else{
    // Large matrix case - not implemented
    printf(" NOT IMPLENTED!");
    throw;
  }
  // Calculate compression ratio and update profiler
  ull raw_ptr_bytes = numBlocksB * 1ll * sizeof(EIdType) * csrb_t_.nc;
  profiler::Instance().cpu_data.compress_ratio = raw_ptr_bytes * 1.0 / total_bytes_after_comp_all;
  profiler::Instance().cpu_data.compressd_len = total_bytes_after_comp_all;
  // printf("csra_.ptr%p numBlocksA%d\n", csra_.ptr, numBlocksA);
  // printf("compress time %.4f,convert_vcsr_time %.4f\n", prf.compress_time, prf.convert_vcsr_time);
  // printf("%p %d\n", &csra_, csra_.nnz);
  // Calculate total compression time (max across threads for each block)
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
  profiler::Instance().cpu_data.throughput_kBpS = raw_ptr_bytes / total_millisecs;
}

}//namespace CESpGEMM