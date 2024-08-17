#include"FileIO.h"
#include<sys/mman.h>
#include<unistd.h>
#include"profiler.h"

namespace CESpGEMM
{
namespace IO
{
template<typename EIdType, typename ValType>
FileWriter<EIdType,ValType>::FileWriter(const std::string&name){
  m_fp_ptr = fopen((name+".ptr").c_str(), "wb+");
  m_fp_idx = fopen((name+".idx").c_str(), "wb+");
  m_fp_val = fopen((name+".val").c_str(), "wb+");
}

template<typename EIdType, typename ValType>
void FileWriter<EIdType,ValType>::write_block_csr(IdxType N, EIdType*ptr_start, IdxType*idx_start, ValType*val_start){
  EIdType nnz=ptr_start[N]-ptr_start[0];
  fseek(m_fp_idx, 0, SEEK_SET);
  fseek(m_fp_val, 0, SEEK_SET);
  fwrite(ptr_start, sizeof(EIdType), N, m_fp_ptr);
  fwrite(idx_start, sizeof(IdxType), nnz, m_fp_idx);
  fwrite(val_start, sizeof(ValType), nnz, m_fp_val);
  profiler::Instance().kbytes_io += (sizeof(EIdType)*N+(sizeof(IdxType)+sizeof(ValType))*nnz)/1024.;
}

template<typename EIdType, typename ValType>
void FileWriter<EIdType,ValType>::write_single_ptrval(EIdType*lastPtr){
  fwrite(lastPtr, sizeof(EIdType), 1, m_fp_ptr);
}

template<typename EIdType, typename ValType>
FileWriter<EIdType, ValType>::~FileWriter(){
  printf("!close fwrite\n");
  fclose(m_fp_ptr);
  fclose(m_fp_idx);
  fclose(m_fp_val);
}


} // namespace IO
} // namespace CESpGEMM
