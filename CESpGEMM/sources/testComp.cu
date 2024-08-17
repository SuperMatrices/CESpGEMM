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
#include"helper.cuh"
#include<omp.h>

using namespace std;


using namespace CESpGEMM;

void wrong_info(int pos, int expected, int got){
  // printf("wrong at %d, expected %d, got %d\n", pos, expected, got);
}
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

template<typename EIdType, typename ValType>
bool validate_decompress4B_on_gpu(const csr<EIdType, ValType> &c, const comp_4b_type & cmp){
  uint32_t *d_anc;
  uint8_t *d_data, *d_control;
  uint32_t *d_target;
  int nSegs = cmp.num_segs;
  int nBytes = cmp.bytes_of_data;
  int nNzValues = cmp.num_values;
  int nAll = c.nr + 1;
  EIdType * target = new EIdType[nAll];

  CHK_CUDA_ERR(cudaMalloc((void**)&d_anc, sizeof(int) * 3 * (nSegs+1)));
  CHK_CUDA_ERR(cudaMalloc((void**)&d_data, sizeof(uint8_t) * (nBytes+512)));
  CHK_CUDA_ERR(cudaMalloc((void**)&d_control, sizeof(uint8_t) * (nNzValues+64)));
  CHK_CUDA_ERR(cudaMalloc((void**)&d_target, sizeof(int) * nAll));
  CHK_CUDA_ERR(cudaMemset(d_target, -1, sizeof(uint32_t) * nAll));
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
	size_t compressed_data = cmp.get_compressed_size();
	size_t data_size_of_ptr = nAll*4;
	printf("compress rate=%.4f\n", 1.0*data_size_of_ptr/compressed_data);
  CHK_CUDA_ERR(cudaMemcpy(d_control, cmp.control, sizeof(uint8_t)*cmp.nbytes_control, 
  cudaMemcpyHostToDevice));
  
  decomp4B_knl<512, EIdType><<<nSegs, 512>>>(
    d_anc, d_data, d_control, d_target
  );
  CHK_CUDA_ERR(cudaDeviceSynchronize());
  CHK_CUDA_ERR(cudaGetLastError());
  CHK_CUDA_ERR(cudaMemcpy(target, d_target, sizeof(EIdType)*nAll, cudaMemcpyDeviceToHost));
  bool result = check(nAll, c.ptr, target);
  CHK_CUDA_ERR(cudaFree(d_anc));
  CHK_CUDA_ERR(cudaFree(d_data));
  CHK_CUDA_ERR(cudaFree(d_control));
  CHK_CUDA_ERR(cudaFree(d_target));
  delete[]target;
  return result;
}

template<typename SrcType, typename EIdType, typename ValType, typename Compressor_t, typename Func>
void testComp(const std::string &name, Func &&validate, IdxType blocksizeB, int reps){
  cudaSetDevice(3);
  using namespace std::chrono;
  IO::MMapReader<SrcType, ValType> fr(name);
  coo<SrcType, ValType> co;
  csr<SrcType, ValType> cscB = fr.preprocess(true, co);
  co.free();
  fr.clear();
  printf("csr_fetched!\n");
  
  IdxType N = cscB.nc;
  IdxType numBlocksB = (cscB.nr + blocksizeB - 1)/blocksizeB;
  EIdType aux_len = N+1;
  const int nworkers=8;
  IdxType*aux = new IdxType[ (N+1) * 1ll * (nworkers) ];
  std::atomic_int shared_idx=0;
  std::vector<compress_t> v_comp_ptr(numBlocksB);

  uint32_t*d_anchor, *d_target;
  uint8_t *d_data, *d_control;
  CHK_CUDA_ERR( cudaMalloc(&d_anchor, sizeof(uint32_t)*3*((N+1)/512+10)));
  CHK_CUDA_ERR( cudaMalloc(&d_data, 2*(N+1)));
  CHK_CUDA_ERR( cudaMalloc(&d_control, (N+1)));
  CHK_CUDA_ERR( cudaMalloc(&d_target, sizeof(uint32_t)*(N+1)));
  
  ull total_bytes_after_comp_all=0;

  #pragma omp parallel num_threads(nworkers) shared(shared_idx)
  {
    int tid = omp_get_thread_num();
    ull total_bytes_after_comp = 0;
    for(;;){
      int i = shared_idx.fetch_add(1);
//	printf("??%d\n", tid);
      bool should_break = shared_idx.load()>=numBlocksB;
      if(i<numBlocksB){
        csr<EIdType, ValType> p_csrB = convert_from_csc_to_vector_csr_get_slice<EIdType, EIdType, ValType>(cscB, i * blocksizeB, std::min((i+1)*blocksizeB, cscB.nr) );
				auto &comp_i=v_comp_ptr.at(i);
        compress_ptr<EIdType, ValType, compress_t>(N, p_csrB.ptr, comp_i, aux + tid*(N+1));
      }
      if(should_break) break;
    }
  }

	for(auto &comp_i:v_comp_ptr){
		ull bytes_after_comp = (comp_i.num_segs+1ll) * 3ll * sizeof(int) + ((ull)comp_i.bytes_of_data + comp_i.nbytes_control) ;
		total_bytes_after_comp_all += bytes_after_comp;
	}
  delete[]aux;
  for(int i=0;i<numBlocksB;i++){
		std::cout<<i<<"started"<<std::endl;
    const compress_t&comp_i = v_comp_ptr.at(i);
    CHK_CUDA_ERR( cudaMemcpy(d_anchor, comp_i.anchor_data, sizeof(uint32_t)*3*(comp_i.num_segs+1ll), cudaMemcpyHostToDevice) );
    CHK_CUDA_ERR( cudaMemcpy(d_data, comp_i.data, 1*comp_i.bytes_of_data, cudaMemcpyHostToDevice) );
    CHK_CUDA_ERR( cudaMemcpy(d_control, comp_i.control, 1*comp_i.nbytes_control, cudaMemcpyHostToDevice) );
    for(int _=0;_<reps;_++){
      decomp_knl<512,uint32_t><<<comp_i.num_segs, 512>>>(d_anchor,d_data,d_control, d_target);
    }
    cudaDeviceSynchronize();
	  std::cout<<i<<"finished"<<std::endl;
  }
  ull total_h2d = total_bytes_after_comp_all * reps;
  printf("total_h2d:%lld\nsingle:%lld\n", total_h2d, total_bytes_after_comp_all);
}

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

int main(int argc, char **argv){
  using uint = uint32_t;
  using ull = unsigned long long;
  // test_read(argv[1]);
  string path=argv[1];
  int sizeB= atol(argv[2]);
  int reps= atol(argv[3]);
  auto [rows, cols, nnz] = IO::FileReader<ull,float>::readBanner(path);

  printf("(row,col,nnz)=(%d,%d,%lld)\n", rows, cols, nnz);
  if(nnz<1e9){
    printf("!!!processing SMALL matrix\n");
    testComp<uint, uint, float, compress_t>(path, validate_decompress4B_on_gpu<uint,float>, sizeB, reps);
  }
  else{
    printf("!!!processing BIG matrix\n");
    // testComp<ull, uint, float, comp_4b_type>(path, validate_decompress4B_on_gpu<uint, float>, sizeB, reps);
  }
  

  return 0;
}
