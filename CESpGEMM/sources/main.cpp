#include"FileIO.h"
#include"mmap_read.h"
#include"CSR.h"
#include"Storage.h"
// #include"TaskManage.h"
#include<queue>
#include<thread>
#include<map>
#include"prepare.h"
#include"compute-dispatch.h"
#include"pipelined-scheme.h"
#include"profiler.h"
#include<iomanip>
#include<fstream>

using namespace CESpGEMM;
using std::string;


struct X{
  X(){puts("X()");}
  ~X(){puts("~X()");}
  X(X&&){
    puts("X(&&)");
  }
  X(const X&){
    puts("X(c&)");
  }
} ;

template<typename SrcType, typename EIdType, typename ValType>
void solve(const string&pathA, const string&pathB, bool a_equal_b, bool ata, int blockSizeA, int blockSizeB, ull poolsize, int nworkers, bool file_write, std::string result_path, double gpu_friend_ratio, bool just_init){
  printf("solve : %d, %d\n", sizeof(SrcType), sizeof(EIdType));
  IO::MMapReader<SrcType, ValType> fr(pathA);
  coo<SrcType, ValType> cooa, coob;
  csr<SrcType, ValType> csra = fr.preprocess(0, cooa);
  if(a_equal_b) coob = std::move(cooa);
  cooa.free();

  bool b_trans = a_equal_b && ata;
  bool b_trans_final = b_trans ^ 1;
  fr.reopen(pathB);
  csc<SrcType, ValType> cscb = fr.preprocess(b_trans_final, coob);
  coob.free();
  bool single_block=0;
  
  std::vector<int>nz_per_row(cscb.nc) ;
  for(int i=0;i<cscb.nnz;i++){
    nz_per_row[cscb.idx[i] ] ++;
  }
  
  std::vector<ull> block_flops((csra.nr+blockSizeA-1)/blockSizeA);
  for(int i=0;i<csra.nr;i++){
    ull cur_flops = 0;
    for(int j=csra.ptr[i];j<csra.ptr[i+1];j++){
      cur_flops += nz_per_row [csra.idx[j]] ;
    }
    block_flops[i/blockSizeA] += cur_flops;
  }

  std::vector<ull>block_flops_ord=block_flops;
  std::sort(block_flops_ord.begin(), block_flops_ord.end(), std::greater<ull>());
  int gpu_thresh_idx = (int)(block_flops_ord.size() * gpu_friend_ratio);
  ull gpu_thresh = block_flops_ord.at( gpu_thresh_idx );
  if constexpr(sizeof(EIdType) == 4){
    ull max_flops = block_flops_ord[0];
    printf("max_flops = %lld\n", max_flops);
    if(max_flops*3/*2buffer+extra*/*2/*float+int*/ + 256ll * cscb.nr + cscb.nnz*2ll + cscb.nc * 2ll < 15ll*1024/4*1024*1024ll){
      single_block = true;
    }
  }

  single_block=false;

  if(single_block){
    blockSizeB = cscb.nr;
    printf("Change BlockSizeB to %d\n", blockSizeB);
  }
  
  CHK_ASSERT(csra.nc == cscb.nc); // cscb.nc = csrb.nr
  printf("csra=%p, getting b\n", &csra);
  int numBlocksB = (cscb.nr + blockSizeB - 1) / blockSizeB;
  
  printf("%d %d %lld\n", csra.nr, csra.nc, csra.nnz);
  IdxType nbA=(csra.nr+blockSizeA-1)/blockSizeA;
  
  printf("before gs\n");
  
  GlobalStorage<SrcType, EIdType, ValType>::Init(blockSizeA, blockSizeB, nbA, numBlocksB, poolsize, nworkers, std::move(csra), std::move(cscb), std::move(block_flops), gpu_thresh, file_write);
  if(just_init) return;

  if(single_block){
    PIPE_LINE::do_works<SrcType,EIdType, ValType,true>(GlobalStorage<SrcType, EIdType, ValType>::Instance(), file_write, result_path);
  }else{
    PIPE_LINE::do_works<SrcType,EIdType, ValType,false>(GlobalStorage<SrcType, EIdType, ValType>::Instance(), file_write, result_path);
  }

}

template <typename T>
void printval(const char*name, T val){
  
  std::cout<<std::fixed<<std::setprecision(4);
  std::cout<<name<<":"<<val<<'\n';
}



int main(int argc, char**argv){
  using uint = uint32_t;
  using valType = float;
  using ReaderLL = IO::FileReader<ull, valType>;


  string usage=
  "\tRequired command line arguments:\n\
  \t\t-Path To Matrix A (.mtx). E.g. -A path/to/a.mtx\n\
  \tAdditional command line arguments:\n\
  \t\t-Path To Matrix B (.mtx). E.g. -B path/to/b.mtx\n\
  \t\t-Calculate A^T*A. E.g. -ATA 1\n\
  \t\t-rows in a block of A. E.g. -BA 1024\n\
  \t\t-columns in a block of B. E.g. -BB 1024\n\
  \t\t-number of cpu wokers in openmp. E.g. -NW 8\n\
  \t\t-PoolSize, to contain blocksize results(default 1e8), E.g -POOL 10000000\n\
  \t\t-O, the name of the file of the SpGEMM result(in binary), E.g -O result_a";

  int blockSizeA = 0;
  int blockSizeB = 0;
  size_t poolsize = 1e8;
  int nworkers = 8;
  int ata=false, just_init=false;
  string pathA{}, pathB{};
  string result_path{};
  

  std::map<string, int*>argKeyInt = {
    {"-BA", &blockSizeA},
    {"-BB", &blockSizeB},
    {"-ATA", &ata},
    {"-SKIP", &just_init},
    {"-NW", &nworkers},
  } ;
  
  std::map<string, size_t*>argKeyLL = {
    {"-POOL", &poolsize}
  } ;

  std::map<string, string*>argKeyStr = {
    {"-A", &pathA},
    {"-B", &pathB},
    {"-O", &result_path}
  } ;

  for(int iii=1;iii<argc;iii++){
    string s = argv[iii];
    if(argKeyInt.find(s)!=argKeyInt.end()){
      int* value = argKeyInt[s];
      *value = atoi(argv[++iii]);
    }
    else if(argKeyLL.find(s)!=argKeyLL.end()){
      size_t* value = argKeyLL[s];
      *value = atoll(argv[++iii]);
    }
    else if(argKeyStr.find(s)!=argKeyStr.end()){
      string&tgt = *argKeyStr[s];
      string v = string(argv[++iii]);
      tgt.swap(v);
    }
  }

  bool a_equal_b=false;
  if(pathB.length()==0){
    pathB = pathA;
    a_equal_b = true;
  }

  bool file_write = result_path.size()!=0;
  if(file_write){
    std::ofstream ofs(result_path + ".ptr", std::ios_base::out);
    if( ofs.fail() ){
      printf("failed! opening %s\n", result_path.c_str());
      result_path = "";
    }
  }
  
  bool is_large = 0;
  {
    auto [banner_row, banner_col, banner_nnz] = ReaderLL::readBanner(pathA);
    if(banner_nnz > 1e9) is_large = true;
  }
  {
    auto [banner_row, banner_col, banner_nnz] = ReaderLL::readBanner(pathB);
    if(banner_nnz > 1e9) is_large = true;
    if(blockSizeA == 0 || blockSizeB == 0){
      // blockSizeA=4096;
      blockSizeA=12288;
      blockSizeB=24576;
      // if(banner_col<1'000'000){
      //   blockSizeB = 12288;
      // }
      // else if(banner_col<65'000'000){
      //   blockSizeB = 65535;
      // }else{
      //   blockSizeB = 100'000;
      // }
      printf("Set BlockSize to :%d, %d\n", blockSizeA, blockSizeB);
    }
  }

  {
    printf("!!===================================\n") ;
    std::cout<<pathA<<' '<<pathB<<" "<<blockSizeA<<' '<<blockSizeB<<"!\n";
    std::cout<<"print file ? "<<file_write<<"\n";
    if(file_write) std::cout<<"result path="<<result_path<<"\n";
    printf("!!===================================\n");
  }

  // return 0;


  if(pathA.length()==0){
    std::cout<<usage<<"\n";
    return 0;
  }

  profiler::Init();
  
  double gpu_friend_ratio = 0.03;

#ifdef TEST_PREPROC
  just_init = true;
#endif

  printf("is_large? %s\n", is_large?"YES":"NO");
  if(is_large /*|| true*/){
    solve<ull, uint, float>(pathA, pathB, a_equal_b, ata, blockSizeA, blockSizeB, poolsize, nworkers, file_write, result_path, gpu_friend_ratio, just_init);
  }
  else{
    solve<uint, uint, float>(pathA, pathB, a_equal_b, ata, blockSizeA, blockSizeB, poolsize, nworkers, file_write, result_path, gpu_friend_ratio, just_init);
  }

  profiler &prf = profiler::Instance();
  
#define PRTVAL(x) printval(#x, (x))
  printf("-------preproc&&CPU----\n");
  PRTVAL(prf.compress_time);
  PRTVAL(prf.convert_vcsr_time) ;
  PRTVAL(prf.compress_ratio);
  PRTVAL(prf.compressd_len);
  PRTVAL(prf.cpu_compute_time) ; 
  PRTVAL(prf.throughput_kBpS);
  PRTVAL(prf.skipped_col_block);
  PRTVAL(prf.prepare_time);
  printf("-----------GPU---------\n");
  PRTVAL(prf.h2d_time) ;
  PRTVAL(prf.kernel_time) ;
  PRTVAL(prf.d2h_time) ;
  PRTVAL(prf.gpu_computer_time);
  PRTVAL(prf.kbytes_h2d);
  PRTVAL(prf.kbytes_d2h);
  printf("---------PRINTER-------\n");
  PRTVAL(prf.merge_time) ;
  PRTVAL(prf.io_time) ;
  PRTVAL(prf.kbytes_io);
  printf("---------ALL-----------\n");
  PRTVAL(prf.t_pipeline);
  PRTVAL(prf.blocks_cpu);
  PRTVAL(prf.blocks_gpu);
  printf("--------OTHERS---------\n");
  // FILE*fp=fopen("worker-trace.txt","w+");
  // for(auto &[k,v]: profiler::Instance().worker_trace){
  //   fprintf(fp, "%s:%d\n",  Map_Util::Instance().get((void*)k).c_str(), v.size());
  //   for(auto [l,r]:v){
  //     fprintf(fp, "%lld,%lld.", l,r);
  //   }
  //   fprintf(fp, "\n");
  // }
  // fclose(fp);


#undef PRTVAL
  return 0;
}