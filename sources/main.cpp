#include"FileIO.h"
#include"mmap_read.h"
#include"CSR.h"
#include"Storage.h"
#include<queue>
#include<thread>
#include<map>
#include"prepare.h"
#include"compute-dispatch.h"
#include"pipelined-scheme.h"
#include"profiler.h"
#include<iomanip>
#include<fstream>
#include"autotune.h"
#include"logger.h"
#include"avail_devices.h"
#include"Argparse.h"

using namespace CESpGEMM;
using std::string;


// Debug struct to test move semantics
// debug struct X;
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


// Fetch matrices A and B from files and convert to borrowed pointers
template<typename SrcType, typename ValType>
void fetch_matrix(const string&pathA, const string&pathB, bool a_equal_b, bool ata, shared_ptr<csr<SrcType,ValType>>&a, shared_ptr<csc<SrcType,ValType>>&b){
  IO::MMapReader<SrcType, ValType> fr(pathA);
  coo<SrcType, ValType> cooa, coob;
  csr<SrcType, ValType> csra = fr.preprocess(0, cooa);
  if(a_equal_b) coob = std::move(cooa);
  cooa.free();
  bool b_trans_final = (a_equal_b && ata) ^ 1;
  fr.reopen(pathB);
  csc<SrcType, ValType> cscb = fr.preprocess(b_trans_final, coob);
  coob.free();
  a = csra.make_borrowed();
  b = cscb.make_borrowed();
  return;
}

// Main solve function: load matrices, partition, compute SpGEMM
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
  

  // std::vector<ull> block_flops((csra.nr+blockSizeA-1)/blockSizeA);
  
  // std::vector<int>nz_per_row(cscb.nc) ;
  // for(EIdType i=0;i<cscb.nnz;i++){
  //   nz_per_row[cscb.idx[i] ] ++;
  // }
  
  // for(int i=0;i<csra.nr;i++){
  //   ull cur_flops = 0;
  //   for(int j=csra.ptr[i];j<csra.ptr[i+1];j++){
  //     cur_flops += nz_per_row [csra.idx[j]] ;
  //   }
  //   block_flops[i/blockSizeA] += cur_flops;
  // }

  // Estimate flops for each row block
  std::vector<ull> block_flops = count_flops_util(csra, cscb, blockSizeA, true);
  // Sort flops to determine GPU threshold
  std::vector<ull> block_flops_ord=block_flops;
  std::sort(block_flops_ord.begin(), block_flops_ord.end(), std::greater<ull>());
  // Find flops threshold for GPU assignment
  int gpu_thresh_idx = (int)(block_flops_ord.size() * gpu_friend_ratio);
  ull gpu_thresh = block_flops_ord.at( gpu_thresh_idx );
  // Check if single block mode is feasible (small enough for GPU)
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
  
  // Validate matrix dimensions match
  CHK_ASSERT(csra.nc == cscb.nc); // cscb.nc = csrb.nr
  printf("csra=%p, getting b\n", &csra);
  int numBlocksB = (cscb.nr + blockSizeB - 1) / blockSizeB;
  
  printf("%d %d %lld\n", csra.nr, csra.nc, csra.nnz);
  IdxType nbA=(csra.nr+blockSizeA-1)/blockSizeA;
  
  printf("before gs\n");

  // Compression parameters: segment length, min zero length, max zero heads
  CompInfo cmp(512, 16, 2);

  shared_ptr<csr<SrcType, ValType>> csra_shared = csra.make_borrowed();
  shared_ptr<csc<SrcType, ValType>> cscb_shared = cscb.make_borrowed();
  
  // Initialize global storage with matrices and parameters
  GlobalStorage<SrcType, EIdType, ValType>::Init(blockSizeA, blockSizeB, nbA, numBlocksB, poolsize, nworkers, csra_shared, cscb_shared, std::move(block_flops), gpu_thresh, file_write, cmp);
  if(just_init) return;

  if(single_block){
    PIPE_LINE_1::do_works_mgpu<SrcType,EIdType, ValType, true>(GlobalStorage<SrcType, EIdType, ValType>::Instance(), file_write, result_path, false, {});
  }else{
    PIPE_LINE_1::do_works_mgpu<SrcType,EIdType, ValType, false>(GlobalStorage<SrcType, EIdType, ValType>::Instance(), file_write, result_path, false, {});
  }
}

// Print profiling value with fixed precision
template <typename T>
void printval(const char*name, T val){
  std::cout<<std::fixed<<std::setprecision(4);
  std::cout<<name<<":"<<val<<'\n';
}



// Print all profiling statistics
void print_profiler(){
  profiler &prf = profiler::Instance();
  profiler_mgpu &mprof = profiler_mgpu::Instance();
  
#define PRTVAL(x) printval(#x, (x))
  printf("-------preproc&&CPU----\n");
  PRTVAL(prf.cpu_data.compress_time);
  PRTVAL(prf.cpu_data.convert_vcsr_time) ;
  PRTVAL(prf.cpu_data.compress_ratio);
  PRTVAL(prf.cpu_data.compressd_len);
  PRTVAL(prf.cpu_data.cpu_compute_time) ; 
  PRTVAL(prf.cpu_data.throughput_kBpS);
  PRTVAL(prf.cpu_data.skipped_col_block);
  PRTVAL(prf.cpu_data.prepare_time);
  PRTVAL(prf.cpu_data.sorting_time);
  printf("-----------GPU---------\n");
  PRTVAL(prf.gpu_data->h2d_time) ;
  PRTVAL(prf.gpu_data->kernel_time) ;
  PRTVAL(prf.gpu_data->d2h_time) ;
  PRTVAL(prf.gpu_data->gpu_computer_time);
  PRTVAL(prf.gpu_data->kbytes_h2d);
  PRTVAL(prf.gpu_data->kbytes_d2h);
  for(int i=0;i<mprof.num_devices;i++){
    printf("---------GPU:%d--------\n", i);
    PRTVAL(mprof.gpu_data[i]->h2d_time) ;
    PRTVAL(mprof.gpu_data[i]->kernel_time) ;
    PRTVAL(mprof.gpu_data[i]->d2h_time) ;
    PRTVAL(mprof.gpu_data[i]->gpu_computer_time);
    PRTVAL(mprof.gpu_data[i]->kbytes_h2d);
    PRTVAL(mprof.gpu_data[i]->kbytes_d2h);
  }
  printf("---------PRINTER-------\n");
  PRTVAL(prf.cpu_data.merge_time) ;
  PRTVAL(prf.cpu_data.io_time) ;
  PRTVAL(prf.cpu_data.kbytes_io);
  printf("---------ALL-----------\n");
  PRTVAL(prf.cpu_data.t_pipeline);
  PRTVAL(prf.cpu_data.blocks_cpu);
  PRTVAL(prf.cpu_data.blocks_gpu);
  printf("--------OTHERS---------\n");
}

// Run SpGEMM with given parameters
template<typename SrcType, typename EIdType, typename ValType>
void run_parameter(shared_ptr<csr<SrcType,ValType>> a, shared_ptr<csr<SrcType,ValType>> b, size_t poolsize, int num_workers, bool file_write, int device_id, const CESpGEMM::TuneParam &param){
  using namespace std::chrono;
  using Runner_t = CESpGEMM::GA_Runner<SrcType,EIdType,ValType>;

  Runner_t runner(a, b, poolsize, num_workers, file_write, device_id);
  double exe_time = runner.execute_param(param, true);
  printf("-------------------------------------------------\n");
  print_profiler();
  printf("exe_time=%.3f\n", exe_time);
  printf("-------------------------------------------------\n");
}

// Search for optimal parameters using genetic algorithm or random search
template<typename SrcType, typename EIdType, typename ValType>
void search_parameter( shared_ptr<csr<SrcType,ValType>> a, shared_ptr<csr<SrcType,ValType>> b, size_t poolsize, int num_workers, bool file_write, int device_id, bool random_only=false){
  using namespace std::chrono;
  const int num_population = 14;
  const int num_iter = 8;
  using Runner_t = CESpGEMM::GA_Runner<SrcType, EIdType, ValType>;
  
  Runner_t runner(a, b, poolsize, num_workers, file_write, device_id);
  
  auto t_begin = steady_clock::now();
  CESpGEMM::TuneParam best_param = 

    random_only?
    CESpGEMM::RandomSearch_for_SpGEMM<Runner_t>(num_population * num_iter, runner) :
    CESpGEMM::GenericAlgorithm_for_SpGEMM<Runner_t,GA_Individual<Runner_t>> (num_population, num_iter, runner);
    // CESpGEMM::TuneParam(4,64,512,4096,32768,10);
    // CESpGEMM::TuneParam(4,16,512,2048,65535,2);
    // CESpGEMM::TuneParam(3,2,1024,1024,40000,25);

  // auto t_mid = steady_clock::now();
  // CESpGEMM::TuneParam best_param_rand = 
  //   CESpGEMM::TuneParam(4,64,512,4096,32768,10);
    // CESpGEMM::TuneParam(1,16,256,2048,65535,2);
    // CESpGEMM::TuneParam(3,2,1024,1024,36692,25);
    // CESpGEMM::TuneParam(4,16,512,12288,65535,10);
  
  auto t_end = steady_clock::now();


  printf("==================best param================\n");
  printf("==== %s elapsed time: %.3fms\n", random_only?"RS":"GA", (double)(duration_cast<microseconds>(t_end-t_begin).count()/1000.0));
  best_param.print();
  printf("==================time=========================\n");
  for(int i=0;i<1;i++){
    double exe_time = runner.execute_param(best_param, true);
    printf("-------------------------------------------------\n");
    print_profiler();
    printf("exe_time=%.3f\n", exe_time);
    printf("-------------------------------------------------\n");
  }
}




int main(int argc, char**argv){
  // Parse command line arguments
  using uint = uint32_t;
  using valType = float;
  using ReaderLL = IO::FileReader<ull, valType>;


  // string usage=
  // "\tRequired command line arguments:\n\
  // \t\t-Path To Matrix A (.mtx). E.g. -A path/to/a.mtx\n\
  // \tAdditional command line arguments:\n\
  // \t\t-Path To Matrix B (.mtx). E.g. -B path/to/b.mtx\n\
  // \t\t-Calculate A^T*A. E.g. -ATA 1\n\
  // \t\t-rows in a block of A. E.g. -BA 1024\n\
  // \t\t-columns in a block of B. E.g. -BB 1024\n\
  // \t\t-number of cpu wokers in openmp. E.g. -NW 8\n\
  // \t\t-PoolSize, to contain blocksize results(default 1e8), E.g -POOL 10000000\n\
  // \t\t-RAND, whether to run random only. E.g -RAND 1\n\
  // \t\t-O, the name of the file of the SpGEMM result(in binary), E.g -O result_a";

  int blockSizeA = 0;
  int blockSizeB = 0;
  size_t poolsize = 1e8;
  int nworkers = 8;
  int ata=false, just_init=false;
  int is_param_given = 0;
  int random_only = 0;
  int runtime_debug_level = 0;
  int gmem_limit_GB = -1;

  // Default tuning parameters
  TuneParam given_param(4,16,512,4096,12288,3);
  string pathA{}, pathB{};
  string result_path{};
  string devices_avail{};

  // Parse command line arguments
  ArgParser parser;
  parser.add_reference("-BA", blockSizeA);
  parser.add_reference("-BB", blockSizeB);
  parser.add_reference("-ATA", ata);
  parser.add_reference("-SKIP", just_init);
  parser.add_reference("-NW", nworkers);
  parser.add_reference("-GIVE", is_param_given);
  parser.add_reference("-NZHEAD", given_param.t.max_zero_head);
  parser.add_reference("-ZLEN", given_param.t.min_zero_length);
  parser.add_reference("-SLEN", given_param.t.seglen);
  parser.add_reference("-GRATIO", given_param.t.gpu_ratio_percent);
  parser.add_reference("-RAND", random_only);
  parser.add_reference("-DEBUG", runtime_debug_level);
  parser.add_reference("-MAX_GMEM", gmem_limit_GB);

  parser.add_reference("-POOL", poolsize);
  parser.add_reference("-A", pathA);
  parser.add_reference("-B", pathB);
  parser.add_reference("-O", result_path);
  parser.add_reference("-DEVICES", devices_avail);

  parser.parse(argc, argv);

  if(devices_avail.length() == 0){
    devices_avail = "0";
  }
  
  // Set GPU memory limit if specified
  if(gmem_limit_GB != -1){
    uint64_t max_bytes = gmem_limit_GB * 1024ull * 1024ull * 1024ull;
    GpuComputerAsync<default_gs_t, false>::global_memory_limit = max_bytes;
    GpuComputerAsync<default_gs_t, true>::global_memory_limit = max_bytes;
    GpuComputerAsync<large_gs_t, false>::global_memory_limit = max_bytes;
    GpuComputerAsync<large_gs_t, true>::global_memory_limit = max_bytes;
  }

  // Set which GPUs to use
  GPUSelection::set_devices(devices_avail.c_str());
  CHK_ASSERT_EQL((int)GPUSelection::get_device_map().size(), NUM_DEVICES);

  Logger::set_runtime_debug_level(runtime_debug_level);
  LOG_DEBUG(0, "logger debug level = "<<runtime_debug_level);

  // Apply block sizes to parameters
  given_param.t.sizeA=blockSizeA;
  given_param.t.sizeB=blockSizeB;

  // Check if B matrix is same as A
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
  
  // Determine if matrices are large enough for special handling
  bool is_large = 0;
  {
    // Read matrix A dimensions to check size
    auto [banner_row, banner_col, banner_nnz] = ReaderLL::readBanner(pathA);
    if(banner_nnz > 1e9) is_large = true;
  }
  {
    auto [banner_row, banner_col, banner_nnz] = ReaderLL::readBanner(pathB);
    if(banner_nnz > 1e9) is_large = true;
    // Set default block sizes if not specified
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
      // printf("Set BlockSize to :%d, %d\n", blockSizeA, blockSizeB);
    }
  }

  {
    printf("!!===================================\n") ;
    std::cout<<pathA<<' '<<pathB<<" "<<blockSizeA<<' '<<blockSizeB<<"!\n";
    std::cout<<"print file ? "<<file_write<<"\n";
    if(file_write) std::cout<<"result path="<<result_path<<"\n";
    printf("!!===================================\n");
  }

  if(pathA.length()==0){
    parser.print_usage_and_exit();
  }

  // Get primary GPU device ID
  int devId = GPUSelection::get_device_map().front();
  // Initialize profiler with device
  profiler::Init(devId);
  profiler_mgpu::Init(NUM_DEVICES);
  
  // double gpu_friend_ratio = 0.03;

#ifdef TEST_PREPROC
  just_init = true;
#endif

  printf("is_large? %s\n", is_large?"YES":"NO");


  // Large matrix path: use given parameters
  if(is_large /*|| true*/){
    printf("skipping large matrix................\n");
    shared_ptr<csr<uint,float>> a,b;
    fetch_matrix<uint, float>(pathA, pathB, a_equal_b, ata, a, b);
    // search_parameter<uint,uint,float> (a,b,poolsize,nworkers,file_write,devId);
    try{
      run_parameter<uint,uint,float>(a,b,poolsize,nworkers, file_write, devId, 
        CESpGEMM::TuneParam(
          4, 16, 512, 4096, 12288, 5 
        )
      );
    }catch(std::exception&e){
      std::cout<<e.what()<<"\n";
    }
    // solve<ull, uint, float>(pathA, pathB, a_equal_b, ata, blockSizeA, blockSizeB, poolsize, nworkers, file_write, result_path, gpu_friend_ratio, just_init);
  }
  // Normal path: search for optimal parameters
  else{
    shared_ptr<csr<uint,float>> a,b;
    fetch_matrix<uint, float>(pathA, pathB, a_equal_b, ata, a, b);
    // Check if parameters were provided via command line
    printf("is given ? %d\n", is_param_given);
    if(is_param_given){
      run_parameter<uint,uint,float>(a,b,poolsize,nworkers,file_write,devId,given_param);
    }
    else search_parameter<uint,uint,float>(a, b, poolsize, nworkers, file_write, devId, random_only);
    // solve<uint, uint, float>(pathA, pathB, a_equal_b, ata, blockSizeA, blockSizeB, poolsize, nworkers, file_write, result_path, gpu_friend_ratio, just_init);
  }


#undef PRTVAL
  return 0;
}