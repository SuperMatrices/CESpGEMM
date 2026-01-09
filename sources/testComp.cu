#include<iostream>
#include"compress4B.h"
#include"decomp4B.cuh"
#include"CSR.h"
#include"FileIO.h"
#include"generator.h"
#include"cuda.h"
#include"cuda_runtime.h"
#include"decomp.cuh"
#include"compressor.h"
#include<functional>
#include"mmap_read.h"
#include<omp.h>
#include"helper.cuh"

using namespace std;


using namespace CESpGEMM;

// Debug helper: print mismatch information
void wrong_info(int pos, int expected, int got){
  printf("wrong at %d, expected %d, got %d\n", pos, expected, got);
}
// Check if decompressed result matches original
template<typename T>
bool check(int len, const T*origin_ptr, const T*result){
  int bad = 0;
  auto handle_err=[&](int pos, int expected, int got){
    wrong_info(pos, expected, got);
    bad ++;
  } ;
  for(int i=0;i<len;i++){
    if((int)result[i]==-1){
      if(origin_ptr[i]-origin_ptr[i-1] != 0){
        printf("surrounding ptr: i-1:%d, i:%d, i+1:%d\n", origin_ptr[i-1], origin_ptr[i], origin_ptr[i+1]);
        handle_err(i, origin_ptr[i], result[i]);
      }
    }
    else if((int)result[i] < 0){
      int l = - result[i];
      int v = i>=l ? result[i-l] : 0;
      // printf(">--- at the end of zeroseg, pos=%d, length=%d, value=%d\n", i, l, v);
      // printf("surrounding ptr: i-1:%d, i:%d, i+1:%d ----<\n", origin_ptr[i-1], origin_ptr[i], origin_ptr[i+1]);
      if(v != origin_ptr[i] ){
        handle_err(i, origin_ptr[i], v);
      }
    }
    else{
      if(origin_ptr[i]!=result[i]){
        handle_err(i, origin_ptr[i], result[i]);
      }
    }
  }
  if(bad) return false;
  return true;
}

// Validate GPU decompression by comparing with CPU result
template<typename EIdType, typename ValType>
bool validate_decompress4B_on_gpu(const csr<EIdType, ValType> &c, const comp_4b_type & cmp){
  // Allocate GPU memory for decompression
  uint32_t *d_anc;
  uint8_t *d_data, *d_control;
  uint32_t *d_target;
  int nSegs = cmp.num_segs;
  int nBytes = cmp.bytes_of_data;
  int nNzValues = cmp.num_values;
  int nAll = c.nr + 1;
  EIdType * target = new EIdType[nAll];

  // Allocate and initialize GPU buffers
  CHK_CUDA_ERR(cudaMalloc((void**)&d_anc, sizeof(int) * 3 * (nSegs+1)));
  CHK_CUDA_ERR(cudaMalloc((void**)&d_data, sizeof(uint8_t) * (nBytes+512)));
  CHK_CUDA_ERR(cudaMalloc((void**)&d_control, sizeof(uint8_t) * (nNzValues+64)));
  CHK_CUDA_ERR(cudaMalloc((void**)&d_target, sizeof(int) * nAll));
  CHK_CUDA_ERR(cudaMemset(d_target, -1, sizeof(uint32_t) * nAll));
  // Copy compressed data to GPU
  CHK_CUDA_ERR(cudaMemcpy(d_anc, cmp.anchor_data, sizeof(int)*3*(nSegs+1), cudaMemcpyHostToDevice));
  CHK_CUDA_ERR(cudaMemcpy(d_data, cmp.data, sizeof(uint8_t) * nBytes, cudaMemcpyHostToDevice));
  // for(int i=0;i<cmp.num_values;i++){
  //   int cv = (cmp.control[i/4] >> ((i%4)*2)) & 3;
  //   cout<<cv;
  //   if(cv==1){
  //     cout<<"("<<i<<")";
  //   }
  // }
  // cout<<"!\n";
  // getchar();
  printf("cmp: %p, %p, %p, %d, %d, %d\n", cmp.anchor_data, cmp.data, cmp.control, cmp.num_segs, cmp.num_values, cmp.bytes_of_data);
  printf("dcontrol: %p\n", d_control);
	// Calculate compression ratio
	size_t compressed_data = cmp.get_compressed_size();
	size_t data_size_of_ptr = nAll*4;
	printf("compress rate=%.4f\n", 1.0*data_size_of_ptr/compressed_data);
  CHK_CUDA_ERR(cudaMemcpy(d_control, cmp.control, sizeof(uint8_t)*cmp.nbytes_control, 
  cudaMemcpyHostToDevice));
  size_t dyn_sharedmem = get_sharedsize_data_decomp(512, 2);
  // Launch GPU decompression kernel
  decomp4B_knl<512, EIdType><<<nSegs, 512, dyn_sharedmem, 0>>>(
    2, d_anc, d_data, d_control, d_target
  );
  // Wait for GPU to finish
  CHK_CUDA_ERR(cudaDeviceSynchronize());
  CHK_CUDA_ERR(cudaGetLastError());
  CHK_CUDA_ERR(cudaMemcpy(target, d_target, sizeof(EIdType)*nAll, cudaMemcpyDeviceToHost));
  // Compare decompressed result with original
  bool result = check(nAll, c.ptr, target);
  CHK_CUDA_ERR(cudaFree(d_anc));
  CHK_CUDA_ERR(cudaFree(d_data));
  CHK_CUDA_ERR(cudaFree(d_control));
  CHK_CUDA_ERR(cudaFree(d_target));
  delete[]target;
  return result;
}

// Test compression and decompression on GPU
template<typename SrcType, typename EIdType, typename ValType, typename Compressor_t>
void testComp(const std::string &name, IdxType blocksizeB, CompInfo param, int num_iter, int device_id){
  // Set target GPU device
  cudaSetDevice(device_id);
  using namespace std::chrono;
  IO::MMapReader<SrcType, ValType> fr(name);
  coo<SrcType, ValType> co;
  csr<SrcType, ValType> cscB = fr.preprocess(true, co);
  co.free();
  fr.clear();
  printf("csr_fetched!\n");
  
  // Get matrix dimensions
  IdxType N = cscB.nc;
  IdxType numBlocksB = (cscB.nr + blocksizeB - 1)/blocksizeB;
  // Allocate auxiliary arrays for compression
  EIdType aux_len = N+1;
  const int nworkers=32;
  IdxType*aux = new IdxType[ (N+1) * 1ll * (nworkers) ];
  std::atomic_int shared_idx=0;
  std::vector<compress_t> v_comp_ptr(numBlocksB, compress_t(param.seglen));

  // Allocate GPU memory for decompression
  uint32_t*d_anchor, *d_target;
  uint8_t *d_data, *d_control;
  CHK_CUDA_ERR( cudaMalloc(&d_anchor, sizeof(uint32_t)*3*((N+1)/512+10)));
  CHK_CUDA_ERR( cudaMalloc(&d_data, 2*(N+1)));
  CHK_CUDA_ERR( cudaMalloc(&d_control, (N+1)));
  CHK_CUDA_ERR( cudaMalloc(&d_target, sizeof(uint32_t)*(N+1)));
  
  // ull total_bytes_after_comp_all=0;

  // Parallel compression of matrix B blocks
  #pragma omp parallel num_threads(nworkers) shared(shared_idx)
  {
    int tid = omp_get_thread_num();
    printf("tid=%d\n",tid);
    for(;;){
      int i = shared_idx.fetch_add(1);
      bool should_break = shared_idx.load()>=numBlocksB;
      if(i<numBlocksB){
        // Extract and compress each column block
        csr<EIdType, ValType> p_csrB = convert_from_csc_to_vector_csr_get_slice<EIdType, EIdType, ValType>(cscB, i * blocksizeB, std::min((i+1)*blocksizeB, cscB.nr) );
        auto &comp_i=v_comp_ptr.at(i);
        
        compress_ptr_parameterized<EIdType, ValType, compress_t>(N, p_csrB.ptr, comp_i, aux + tid*(N+1), param.min_zero_num, param.max_zero_seg);
      }
      if(should_break) break;
    }
  }


  delete[]aux;

  // Test decompression for each block
  for(int i=0;i<numBlocksB;i++){
    std::cout<<i<<"started"<<std::endl;
    const compress_t&comp_i = v_comp_ptr.at(i);
    CHK_CUDA_ERR( cudaMemcpy(d_anchor, comp_i.anchor_data, sizeof(uint32_t)*3*(comp_i.num_segs+1ll), cudaMemcpyHostToDevice) );
    CHK_CUDA_ERR( cudaMemcpy(d_data, comp_i.data, 1*comp_i.bytes_of_data, cudaMemcpyHostToDevice) );
    CHK_CUDA_ERR( cudaMemcpy(d_control, comp_i.control, 1*comp_i.nbytes_control, cudaMemcpyHostToDevice) );
    for(int iter=1;iter<=num_iter;iter++){
      // Run decompression multiple times
      decomp_launch(d_anchor, d_data, d_control, d_target, 512, 2, comp_i.num_segs, 0);
    }
      // decomp_knl<512,uint32_t><<<comp_i.num_segs, 512>>>(d_anchor,d_data,d_control, d_target);
    cudaDeviceSynchronize();
    std::cout<<i<<"finished"<<std::endl;
  }

  // ull total_h2d = total_bytes_after_comp_all * reps;
  // printf("total_h2d:%lld\nsingle:%lld\n", total_h2d, total_bytes_after_comp_all);
}





// Check if two COO matrices are equal
template<typename EIdType, typename ValType>
bool checkCooAllEqual(const coo<EIdType, ValType>&a, const coo<EIdType, ValType>&b){
  if(a.nr != b.nr) return false;
  if(a.nc != b.nc) return false;
  if(a.nnz != b.nnz) return false;
  for(int i=0;i<a.nnz;i++){
    if(a.row[i]!=b.row[i]) return false;
    if(a.col[i]!=b.col[i]) return false;
    if(fabs(a.val[i]-b.val[i])>1e-5) return false;
  }
  return true;
}

// Test and compare MMapReader vs FileReader
void test_read(const string &name){
  using namespace std::chrono;
  auto t0 = std::chrono::steady_clock::now();
  coo<ull,float> c,d;
  {
    IO::MMapReader<ull, float> rd(name.c_str());
    c = rd.read_matrix_as_coo();
    cout<<c.nr<<' '<<c.nc<<' '<<c.nnz<<"!\n";
    cout<<c.row<<' '<<c.col<<' '<<c.val<<"!\n";
  }
  auto t1 = std::chrono::steady_clock::now();
  {
    IO::FileReader<ull, float> rd(name);
    d = rd.read_matrix_as_coo();
    cout<<c.nr<<' '<<c.nc<<' '<<c.nnz<<"!\n";
    cout<<c.row<<' '<<c.col<<' '<<c.val<<"!\n";
  }
  auto t2 = std::chrono::steady_clock::now();
  
  cout<<"mmap time: "<<duration_cast<microseconds>(t1-t0).count()/1000.0<<"\n";
  cout<<"fread time: "<<duration_cast<microseconds>(t2-t1).count()/1000.0<<"\n";
  cout<<"equal? "<<(checkCooAllEqual(c,d)?"Yes":"No")<<"\n";

}

// Test CPU compression and GPU decompression
template<typename SrcType, typename EIdType, typename ValType>
void test_comp_decomp(const csr<SrcType,ValType> &mat, CompInfo param){
  // Only supports 32-bit indices
  static_assert(sizeof(SrcType) == 4);
  using comp_t = Compressor<IdxType>;
  // Compress ptr array
  comp_t cmp(param.seglen);
  IdxType *aux = new IdxType[mat.nr+1];
  compress_ptr_parameterized<SrcType, ValType>(mat.nr, mat.ptr, cmp, aux, param.min_zero_num, param.max_zero_seg);
  size_t bytes_anchor = sizeof(uint32_t) * 3* (cmp.num_segs+1);
  size_t bytes_data = sizeof(uint8_t) * cmp.bytes_of_data;
  size_t bytes_control = sizeof(uint8_t)*cmp.nbytes_control;
  size_t bytes_target = sizeof(uint32_t) * (mat.nr+1);
  // Allocate GPU memory for decompression
  uint32_t* d_anchor, *d_target;
  uint8_t* d_data, *d_control;
  uint32_t *h_ptr = new uint32_t[mat.nr+1];
  CHK_CUDA_ERR(cudaMalloc(&d_anchor, bytes_anchor));
  CHK_CUDA_ERR(cudaMalloc(&d_data, bytes_data));
  CHK_CUDA_ERR(cudaMalloc(&d_control, bytes_control));
  CHK_CUDA_ERR(cudaMalloc(&d_target, bytes_target));
  CHK_CUDA_ERR(cudaMemset(d_target, -1, bytes_target));
  CHK_CUDA_ERR(cudaMemcpy(d_anchor, cmp.anchor_data, bytes_anchor, cudaMemcpyHostToDevice));
  CHK_CUDA_ERR(cudaMemcpy(d_data, cmp.data, bytes_data,cudaMemcpyHostToDevice));
  CHK_CUDA_ERR(cudaMemcpy(d_control, cmp.control, bytes_control,cudaMemcpyHostToDevice));
  
  // Launch GPU decompression
  decomp_launch(d_anchor, d_data, d_control, d_target, param.seglen, param.max_zero_seg, cmp.num_segs, 0);
  CHK_CUDA_ERR(cudaPeekAtLastError());
  CHK_CUDA_ERR(cudaMemcpy(h_ptr, d_target, bytes_target,cudaMemcpyDeviceToHost));

  check(mat.nr+1, mat.ptr, h_ptr);

  delete[]aux;
  delete[]h_ptr;
  CHK_CUDA_ERR(cudaFree(d_target));
  CHK_CUDA_ERR(cudaFree(d_anchor));
  CHK_CUDA_ERR(cudaFree(d_data));
  CHK_CUDA_ERR(cudaFree(d_control));
}


int main(int argc, char **argv){
  // Parse command line arguments
  using uint = uint32_t;
  using ull = unsigned long long;
  if(7>=argc){
    printf("%s <path> <blockSizeB> <seglen> <min_zero_len> <max_zero_head> <num_iter> <device_id>\n", argv[0]);
    exit(1);
  }
  // test_read(argv[1]);
  string path=argv[1];
  int sizeB= atol(argv[2]);
  int seglen = atol(argv[3]);
  int min_zero_len = atol(argv[4]);
  int max_zero_head = atol(argv[5]);
  int num_iter = atol(argv[6]);
  int device_id = atol(argv[7]);


  auto [rows, cols, nnz] = IO::FileReader<ull,float>::readBanner(path);
  
  printf("(row,col,nnz)=(%lu,%lu,%lu)\n", rows, cols, nnz);
  printf("!!!processing SMALL matrix\n");
  CompInfo param(seglen, min_zero_len, max_zero_head);
  using comp_t = Compressor<IdxType>;

  testComp<IdxType, IdxType, float, comp_t>(path, sizeB, param, num_iter, device_id);

// template<typename SrcType, typename EIdType, typename ValType, typename Compressor_t, typename Func>
// void testComp(const std::string &name, IdxType blocksizeB, CompInfo param, int num_iter, int device_id){

  return 0;
}
