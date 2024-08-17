#pragma once
#include<cuda.h>
#include<cuda_runtime.h>
#include<iostream>
#include<chrono>
#include<semaphore.h>
#include<map>
#include<thread>
#include<sstream>

namespace CESpGEMM{

class helper{
public:
  static void my_check_error_info(cudaError_t code, const char*func, const char* file, int line){
  if (code != cudaSuccess){
      std::stringstream ss;
      ss << "CUDA Runtime Error at: " << file << ":" << line
                << '\n';
      ss << cudaGetErrorString(code) << " " << func << '\n';
      std::cerr<<ss.str();
      throw;
      // exit(-1);
      // We don't exit when we encounter CUDA errors in this example.
      // std::exit(EXIT_FAILURE);
    }
  }

  static void my_assert(int val, const char*func, const char* file, int line){
    if(val == 0){
      std::stringstream ss;
      ss <<"assertion failed at: "<<file << ":" << line
                  << '\n';
      ss << func << '\n';
      std::cerr<<ss.str();
      throw;
    }
  }
  
  template<typename T>
  static void my_assert_less(T lhs, T rhs, const char* file, int line){
    if(lhs >= rhs){
      std::stringstream ss;
      ss <<"assertion failed at: "<<file << ":" << line
                  << '\n';
      ss << "lhs = "<<lhs<<", \n";
      ss << "rhs = "<<rhs<<". \n";
      std::cerr<<ss.str();
      throw;
    }
  }

  template<typename T>
  static void my_assert_eql(T lhs, T rhs, const char* file, int line){
    if(lhs != rhs){
      std::stringstream ss;
      ss <<"assertion failed at: "<<file << ":" << line
                  << '\n';
      ss << "lhs = "<<lhs<<", \n";
      ss << "rhs = "<<rhs<<". \n";
      std::cerr<<ss.str();
      throw;
    }
  }

  template<typename T>
  static void print_gpu_arr(T*d, int len, int show_l, int show_r, const char*info){
    T*h=new T[len];
    cudaMemcpy(h, d, sizeof(T)*len, cudaMemcpyDeviceToHost);
    for(int i=show_l;i<show_r;i++){
      printf("i %d: %s %d\n", i, info, h[i]);
    }
    delete[]h;
  }


} ;

#define SHOW_GPU_ARR_SEG(arr, len, left, right) ::CESpGEMM::helper::print_gpu_arr((arr), (len), (left), (right), #arr)
#define CHK_CUDA_ERR(val) ::CESpGEMM::helper::my_check_error_info((val), #val, __FILE__, __LINE__)
#define CHK_ASSERT(val) ::CESpGEMM::helper::my_assert((val), #val, __FILE__, __LINE__)
#define CHK_ASSERT_LESS(lhs, rhs) ::CESpGEMM::helper::my_assert_less((lhs), (rhs), __FILE__, __LINE__)
#define CHK_ASSERT_EQL(lhs, rhs) ::CESpGEMM::helper::my_assert_eql((lhs), (rhs), __FILE__, __LINE__)

}

template<typename F>
float timeit(F &&f){
  auto t0 = std::chrono::steady_clock::now();
  f();  
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
}


struct GpuEventTick{
  cudaEvent_t e;
  GpuEventTick(){
    cudaEventCreate(&e);
    cudaEventRecord(e);
  }
  void sync(){
    cudaEventSynchronize(e);
  }
  ~GpuEventTick(){
    cudaEventDestroy(e);
  }
  friend float operator-(const GpuEventTick& e1, const GpuEventTick& e2) {
    float ret = 114514;
    cudaEventElapsedTime(&ret, e1.e, e2.e);
    return ret;
  }

} ;

inline void helper_sem_wait(sem_t*s, const char*help){
  printf(" semwait at %s, %p\n", help, s);
  sem_wait(s);
}
inline void helper_sem_post(sem_t*s, const char*help){
  printf("sempost at %s, %p\n", help, s);
  sem_post(s);
}

class Map_Util{
private:
  std::map<void*,std::string> _name_tb;  
  Map_Util(){

  }
public:
  static Map_Util &Instance(){
    static Map_Util mp{};
    return mp;
  }
  void add(void* p, std::string s){
    _name_tb[p]=s;
  }
  std::string get(void* p){
    return _name_tb[p];
  }
} ;


#ifdef DEBUG

#define SEM_WAIT(U) helper_sem_wait(U, #U)
#define SEM_POST(U) helper_sem_post(U, #U)

#else
#define SEM_WAIT(U) sem_wait(U)
#define SEM_POST(U) sem_post(U)

#endif
