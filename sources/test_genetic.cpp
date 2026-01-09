#include"autotune.h"
#include<cstdio>

// Test runner for genetic algorithm (returns negative sum of parameters)
struct Test_Runner{
  // Mock execution: returns negative sum of parameters (for testing)
  double execute_param(CESpGEMM::TuneParam param)const{
    double s=0;
    for(int i=0;i<5;i++) s-=param.par[i];
    return s;
  }
  
} ;

using namespace std;
int main(){
  // Test genetic algorithm with mock runner
  Test_Runner tr;
  using ga_ind_t = CESpGEMM:: GA_Individual<Test_Runner>;
  // Run genetic algorithm to find optimal parameters
  auto param = CESpGEMM::GenericAlgorithm_for_SpGEMM<Test_Runner, ga_ind_t>(40, 40, tr);
  // Print found parameters
  printf("%d %d %d %d %d\n", 
    param.t.max_zero_head, param.t.min_zero_length, param.t.seglen, param.t.sizeA, param.t.sizeB);
  return 0;
}
