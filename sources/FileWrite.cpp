#include"FileIO.h"
#include<sys/mman.h>
#include<unistd.h>
#include"profiler.h"

namespace CESpGEMM
{
namespace IO
{

template
void FileWriter::write_single_ptrval<unsigned long long>(unsigned long long*);

template
void FileWriter::write_block_csr<unsigned int, unsigned long long, float>(unsigned int, unsigned long long*, unsigned int*, float*);


// FileWriter: Writes CSR matrix blocks to separate binary files (.ptr, .idx, .val)
FileWriter::FileWriter(const std::string&name){
  m_fp_ptr = fopen((name+".ptr").c_str(), "wb+");
  m_fp_idx = fopen((name+".idx").c_str(), "wb+");
  m_fp_val = fopen((name+".val").c_str(), "wb+");
}

// Write a CSR block to binary files (ptr, idx, val)
// N: number of rows, ptr_start: row pointer array, idx_start: column indices, val_start: values
template<typename VertType, typename EdgeType, typename ValType>
void FileWriter::write_block_csr(VertType N, EdgeType*ptr_start, VertType*idx_start, ValType*val_start){
  EdgeType nnz=ptr_start[N]-ptr_start[0];
  fseek(m_fp_idx, 0, SEEK_SET);
  fseek(m_fp_val, 0, SEEK_SET);
  fwrite(ptr_start, sizeof(EdgeType), N, m_fp_ptr);
  fwrite(idx_start, sizeof(VertType), nnz, m_fp_idx);
  fwrite(val_start, sizeof(ValType), nnz, m_fp_val);
  profiler::Instance().cpu_data.kbytes_io += (sizeof(EdgeType)*N+(sizeof(VertType)+sizeof(ValType))*nnz)/1024.;
}

// Write the last pointer value to complete the ptr array
template<typename EdgeType>
void FileWriter::write_single_ptrval(EdgeType*lastPtr){
  fwrite(lastPtr, sizeof(EdgeType), 1, m_fp_ptr);
}

FileWriter::~FileWriter(){
  // printf("!close fwrite\n");
  fclose(m_fp_ptr);
  fclose(m_fp_idx);
  fclose(m_fp_val);
}


} // namespace IO
} // namespace CESpGEMM
