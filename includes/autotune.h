#pragma once
#include<cstdio>
#include<iostream>
#include<string>
#include"CSR.h"
#include<algorithm>
#include"Storage.h"
#include "pipelined-scheme.h"
#include <random>
#include<sstream>
// 使用遗传算法
// 需要一个执行器，输入参数，输出fitness函数
//


// Requires:
// ParameterForm : 
// Code : Array
// Encoder: Param->Code
// Decoder: Code->Param
// 
// #gene order: max_zero_head min_zero_length seglen sizeA sizeB
// #gene range:
// #max_zero_head 1 < max_zero_head < 4
// #min_zero_length 1 < min_zero_length < 256 (power of 2)
// #seglen: 32 < seglen < 1024 (power of 2)
// #sizeA: 4096 < sizeA < 65535 (power of 2)
// #sizeB: 4096 < sizeB < 65535 (power of 2)

// max_zero_head [1..4] 2bit
// min_zero_length [2^1, 2^8] 1..8 3bit
// seglen: [2^7..2^10] 7..10 2bit
// blockSizeA, blockSizeB : [2^10..2^16] 10..16 // 3bit
// total: 2+3+2+3+3 = 13bit
namespace CESpGEMM{



inline int randgen(int l, int r){
  static std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  // static std::mt19937 gen(20250320);
  return gen()%(r-l+1)+l;
}


struct TuneParam {
  struct tp{
    int max_zero_head;
    int min_zero_length;
    int seglen;
    int sizeA;
    int sizeB;
    int gpu_ratio_percent;
  } ;
  union{
    int par[6];
    tp t;
  } ;
  TuneParam(){
    t.max_zero_head = randgen(1,4);
    t.min_zero_length = 1<<randgen(1,8);
    t.seglen = 1<<randgen(7,10);
    t.sizeA = 1<<randgen(10,16);
    t.sizeB = 1<<randgen(13,16);
    t.gpu_ratio_percent = randgen(0,25);
  }

  TuneParam(int maxZeroHead, int minZeroLength, int segLen, int aSize, int bSize, int gpu_ratio_percent){
    t.max_zero_head = maxZeroHead;
    t.min_zero_length = minZeroLength;
    t.seglen = segLen;
    t.sizeA = aSize;
    t.sizeB = bSize;
    t.gpu_ratio_percent = gpu_ratio_percent;
  }
  void mutate(){
    switch(randgen(0,4)){
      case 0: t.max_zero_head = randgen(1,4); break;
      case 1: t.min_zero_length = 1<<randgen(1,8); break;
      case 2: t.seglen = 1<<randgen(7,10); break;
      case 3: t.sizeA = 1<<randgen(10,16); break;
      case 4: t.sizeB = 1<<randgen(13,16); break;
      case 5: t.gpu_ratio_percent = randgen(0,25); break;
    }
  } 
  void cross_over(TuneParam&other){
    int start = randgen(0,4);
    for (int i = start; i < 5; ++i) {
      std::swap(par[i], other.par[i]);  // 交换 par 数组元素
    }
  }

  void print(){
    printf("(%d,%d,%d,%d,%d,%d)\n", t.max_zero_head, t.min_zero_length, t.seglen, t.sizeA, t.sizeB, t.gpu_ratio_percent);
  }
  std::string to_string(){
    std::stringstream ss;
    ss<<'('<<t.max_zero_head<<','<<t.min_zero_length<<','<<t.seglen<<','<<t.sizeA<<','<<t.sizeB<<','<<t.gpu_ratio_percent<<')';
    return ss.str();
  }
  bool equals(const TuneParam&x)const{
    for(int i=0;i<5;i++) if(par[i] != x.par[i]) return false;
    return true;
  }
};


// // cut [L,R)
// inline GeneType CutGene(GeneType g, int L, int R){
//   int Len = R-L;
//   return (g>>L) & ((1<<Len) - 1);
// }

// // min(r, max(l, x))
// inline int clamp(int x, int l, int r){
//   return std::min(r, std::max(l, x));
// }

// TuneParam GA_Decode(GeneType gene){
//   GeneType max_zero_head = CutGene(gene, 0, 2) + 1; // 1,2,3,4
//   GeneType min_zero_length = CutGene(gene, 2, 5) + 1; // 1..8 
//   GeneType seglen = CutGene(gene, 5, 7) + 7; // 7..10
//   GeneType blockSizeA = std::min( CutGene(gene, 7, 10) + 10, 16); //10..17 -> 10..16
//   GeneType blockSizeB = std::min( CutGene(gene, 10, 13) + 10, 16); //10..17 -> 10..16

//   int real_min_zero_length = 1<<min_zero_length;
//   int real_seglen = 1<<seglen;
//   int real_sizeA = 1<<blockSizeA;
//   int real_sizeB = 1<<blockSizeB;
//   return TuneParam(max_zero_head, real_min_zero_length, real_seglen, real_sizeA, real_sizeB);
// } ;
// //2 3 2 3 3
// GeneType GA_Encode(TuneParam param){
//   GeneType max_zero_head = param.max_zero_head - 1;
//   GeneType min_zero_len = __builtin_ctz(param.min_zero_length) - 1;
//   GeneType seglen = __builtin_ctz(param.seglen) - 7;
//   GeneType blockSizeA = __builtin_ctz(param.sizeA) - 10;
//   GeneType blockSizeB = __builtin_ctz(param.sizeB) - 10;
//   return max_zero_head | (min_zero_len << 2) | (seglen<<5) | (blockSizeA<<7) | (blockSizeB<<10);
// }


template<typename SrcType, typename EIdType, typename ValType>
struct GA_Runner{
  std::shared_ptr< csr<SrcType, ValType> > csra;
  std::shared_ptr< csc<SrcType, ValType> > cscb;
  size_t poolsize;
  int num_workers;
  bool file_write;
  int device_id;
  const char* output_dir;
  size_t max_gpu_size_bytes;
  using Gs_t = GlobalStorage<SrcType,EIdType,ValType>;

  GA_Runner(
    std::shared_ptr<csr<SrcType, ValType>> csra,
    std::shared_ptr<csc<SrcType, ValType>> cscb,
    size_t poolsize,
    int num_workers,
    bool file_write,
    int device_id,
    size_t max_gpu_size_bytes=(10ull*1024*1024*1024)
  ) : csra(csra), cscb(cscb), poolsize(poolsize), num_workers(num_workers), file_write(file_write), output_dir("auto-tune-tmp.bin"), device_id(device_id), max_gpu_size_bytes(max_gpu_size_bytes){
    
  }

  template<typename T>
  std::vector<T> equal_stride_sample(T n, int k)const{
    std::vector<T> sampled_blocks;
    if(k==1){
      sampled_blocks.push_back((n-1)/2);
      return sampled_blocks;
    }
    T nspaces = k-1, stride = std::max<int>((n-1)/nspaces, 1);
    sampled_blocks.reserve(k);
    T pos = 0;
    for(int i=0; i<k && pos<n; i++, pos+=stride){
      sampled_blocks.push_back(pos);
    }
    return sampled_blocks;
  }
  
  double execute_param(TuneParam param, bool run_full=false)const{
    int blocksizeA = param.t.sizeA;
    int blocksizeB = param.t.sizeB;
    const csr<SrcType,ValType> & csra_ref = *csra.get(); 
    const csr<SrcType,ValType> & cscb_ref = *cscb.get(); 

    int nbA = (csra_ref.nr + blocksizeA-1)/blocksizeA;
    int nbB = (cscb_ref.nr + blocksizeB-1)/blocksizeB;
    // printf("nbA nbB  :%d %d\n", nbA, nbB);
    // constexpr double gpu_friend_ratio = 0.03;
    const double gpu_friend_ratio = param.t.gpu_ratio_percent * 0.01;
    constexpr int max_num_blocks_sampled = 64;
    const int num_blocks = (nbA+14)/15;
    profiler::reset();
    profiler_mgpu::reset();

    //blockflops
    using namespace std::chrono;

    auto t_count_flop = steady_clock::now();

    std::vector<ull> block_flops = count_flops_util(csra_ref, cscb_ref, blocksizeA, true);

    auto t_start_sort = steady_clock::now();
    std::vector<IdxType> indices(block_flops.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&block_flops](int i, int j){
      return block_flops[i] > block_flops[j];
    } );
    //先排序，再按照计算量的大小来采样，再找到原来的块编号
    // std::sort(block_flops.begin(), block_flops.end(), std::greater<ull>());
    auto t_end_sort = steady_clock::now();
    double flop_count_time = duration_cast<microseconds>(t_start_sort-t_count_flop).count()/1000.0;
    
    profiler & prof = profiler::Instance();
    prof.cpu_data.sorting_time = duration_cast<microseconds>(t_end_sort-t_start_sort).count()/1000.0;

    int gpu_thresh_idx = (int)(block_flops.size() * gpu_friend_ratio) ;

    ull gpu_thresh = block_flops[ indices[gpu_thresh_idx] ];
    CompInfo cmp(param.t.seglen, param.t.min_zero_length, param.t.max_zero_head);
    try{
      Gs_t::Init(
        blocksizeA,
        blocksizeB,
        nbA,
        nbB,
        poolsize,
        num_workers,
        csra,
        cscb,
        std::move(block_flops),
        gpu_thresh,
        max_gpu_size_bytes,
        file_write,
        cmp
      ) ;
    } catch (std::exception) {
      printf("catched!\n");
      return 1e10;
    }

    std::vector<IdxType> sampled_blocks;
    t_start_sort = steady_clock::now();
    if(run_full){
      sampled_blocks.resize(nbA);
      std::iota(sampled_blocks.begin(), sampled_blocks.end(), 0);
    }else{
      sampled_blocks = equal_stride_sample<IdxType>(
        nbA,
        std::max<int>(1, std::min<int>(max_num_blocks_sampled, num_blocks))
      ) ;
      for(int i=0;i<(int)sampled_blocks.size();i++){
        int x = sampled_blocks[i] ;
        sampled_blocks[i]=indices[x];
      }
      sort(sampled_blocks.begin(), sampled_blocks.end());
    }
    t_end_sort = steady_clock::now();
    prof.cpu_data.sorting_time += duration_cast<microseconds>(t_end_sort-t_start_sort).count()/1000.0;

    if(run_full){
      printf("running full: size=%lu\n", sampled_blocks.size());
    }
    
    try{
      LOG_DEBUG(0, "do_works_mgpu");
      double time1 = CESpGEMM::PIPE_LINE_1::do_works_mgpu<SrcType,EIdType,ValType,false>(
        Gs_t::Instance(),
        file_write, 
        output_dir,
        true,
        sampled_blocks
      ) ;

      printf("Running Time=%.8f\n", time1);
      printf("Flop Count Time=%.8f\n", flop_count_time);
      printf("Flop Count Ratio=%.8f\n", flop_count_time / time1);
      printf("Sorting Time=%.8f\n", prof.cpu_data.sorting_time);
      printf("Sorting Ratio=%.8f\n", prof.cpu_data.sorting_time / time1);
      return time1 / (param.t.sizeA *  sampled_blocks.size()) * csra_ref.nr;
    } catch(std::exception &e){
      // printf()
      printf("catched@\n");
      printf("%s\n", e.what());
      return 1e10;
    }
    return 1e10;
  }
} ;

template<class Runner>
class GA_Individual{
private:
  double fitness=1e9;//(here means time) the bigger the worse
  TuneParam param;
  // static Runner<SrcType, EIdType, ValType> runner;

public:
  TuneParam getParam(){
    return param;
  }
  double getFitness()const{
    return fitness;
  }
  void cross_over(GA_Individual &x){
    param.cross_over(x.param);
  }
  void mutate(){
    // constexpr int mutation_rate = ;
    if(randgen(0,9) < 2) param.mutate();
  }
  void compute_fitness(const Runner &runner){
    fitness = runner.execute_param(param);
  } 
  bool operator<(const GA_Individual &x)const{
    return fitness < x.fitness;
  }
  void print(){
    param.print();
  }
  std::string to_string(){
    return param.to_string();
  }
  bool equals(const GA_Individual &x)const {
    return param.equals(x.param);
  }
} ;

template<class Runner>
TuneParam RandomSearch_for_SpGEMM(int num_tries, const Runner&runner){
  TuneParam best_par;
  double t_best = runner.execute_param(best_par);
  while(num_tries--){
    TuneParam rand_par;
    double t = runner.execute_param(rand_par);
    if(t<t_best){
      best_par = rand_par;
    }
  }
  return best_par;
}

template<class Runner, class Individual>
TuneParam GenericAlgorithm_for_SpGEMM(const int num_population, int num_iter, const Runner&runner){
  std::vector<Individual> population(num_population);
  const int keep=2;
  std::vector<Individual> record_last_best(keep);
  Individual best;

  best.compute_fitness(runner);
  for(auto&ind : population){
    ind.compute_fitness(runner);
  }
  auto selection = [&population](int k){
    sort(population.begin(), population.end() );
    population.resize(k); 
  } ;
  auto early_stopping = [&record_last_best, &population, keep]()->bool {
    Individual cur_best = population[0];
    bool ret=1;
    for(int i=0;i<keep;i++){
      if(! cur_best.equals(record_last_best[i])) ret=0;
    }
    for(int i=0;i+1<keep;i++){
      record_last_best[i+1] = record_last_best[i];
    }
    record_last_best[0] = cur_best;
    return ret;
  } ;
  
  for(int iter=1;iter<=num_iter;++iter){
    int k=num_population/2;
    selection(k);
    printf("iter %d, best parameter=%s, time=%.3lf\n", iter, population[0].to_string().c_str(), population[0].getFitness());

    if(population[0] < best){
      best = population[0];
    }

    if(early_stopping() && iter>keep){
      printf("EARLY STOPPPING\n");
      printf("iter %d, best parameter=%s, time=%.3lf\n", iter, population[0].to_string().c_str(), population[0].getFitness());
      break;
    } 

    while(population.size()<num_population){
      int i1 = randgen(0, k - 1);
      int i2 = randgen(1, k - 1);
      i2 = (i1+i2) % k;
      auto &p1 = population[i1];
      auto &p2 = population[i2];
      p1.cross_over(p2);
      p1.mutate();
      p2.mutate();
      p1.compute_fitness(runner);
      p2.compute_fitness(runner);
      population.push_back(p1);
      population.push_back(p2);
    }
  }

  int idx = 0;
  for(int i=0;i<(int)population.size();i++){
    if(population[i] < population[idx]) {//beter
      idx = i;
    }
  }

  best.print();
  population[idx].print();
  if(population[idx]<best) best=population[idx];
  return best.getParam();
}

}
