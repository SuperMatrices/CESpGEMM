#include"TaskManage.h"


namespace CESpGEMM
{

void TaskManage::addCPUTask(IdxType rBid, IdxType cBid){
  cq.enque(std::pair{rBid, cBid});
}

void TaskManage::addGPUTask(IdxType rBid, IdxType cBid){
  gq.enque(std::pair{rBid, cBid});
}


void TaskManage::finishAll(){
  _all_finished = true;
  cq.enque(std::pair{FINISH, FINISH});
  gq.enque(std::pair{FINISH, FINISH});
}

bool TaskManage::cpuIsFree(){
  return cq.empty() && !_cpu_running;
}

bool TaskManage::gpuIsFree(){
  return gq.empty() && !_gpu_running;
}

std::pair<IdxType, IdxType> TaskManage::popCPUTask(){
  return cq.pop_trigger_variable(_cpu_running);
}
std::pair<IdxType, IdxType> TaskManage::popGPUTask(){
  return gq.pop_trigger_variable(_gpu_running);
}




} // namespace CESpGEMM
