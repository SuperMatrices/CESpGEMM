#pragma once
#include"CSR.h"
#include<iostream>
#include<cstdio>
// #include<liburing.h>

namespace CESpGEMM
{
namespace IO{


template<typename EIdType, typename ValType>
class FileReader{
private:
struct MyFP{
  FILE*fp;
  char*_buffer,*_st,*_ed;
  MyFP(const std::string&name){
    fp = fopen(name.c_str(), "r");
    if(fp==nullptr){
      throw std::runtime_error("cannot find file!\n");
    }
    _buffer = new char[readBufferLength];
    reload();
  }
  ~MyFP(){
    printf("close file\n");
    delete[]_buffer;
    fclose(fp);
  }
  void reload(){
    _st = _buffer;
    _ed = _st + fread(_buffer, 1, readBufferLength, fp);
  }
  bool check_end(){
    if(_st==_ed) reload();
    if(_st==_ed) return true;
    return false;
  }

  FILE*get(){return fp;}
} ;

public:
  coo<EIdType,ValType> read_matrix_as_coo();
  csr<EIdType,ValType> preprocess(bool transpose, coo<EIdType, ValType>& contain);
  // csr<EIdType,ValType> load_csr(std::string name);
  // void write_csr(std::string name, csr<EIdType,ValType>&);


  // csr<EIdType,ValType>> read_matrix_as_csr(std::string name);
  static std::tuple<size_t,size_t,size_t> readBanner(const std::string &name);
  static constexpr const char* common_csr_suffix = sizeof(EIdType) == 4 ? ".dcsr":".ldcsr";
  FileReader()=delete;
  FileReader(const std::string &name){
    _myfp = std::make_unique<MyFP>(name);
  }
  void reopen(const std::string &name){
    _myfp = std::make_unique<MyFP>(name);
  }
  ~FileReader()=default;

private:
  char gchar();
  constexpr static size_t readBufferLength = 1<<23;
  std::unique_ptr<MyFP> _myfp;
  void getLine(std::string&);
} ;

template class FileReader<IdxType, float>;
template class FileReader<ull, float>;


template<typename EIdType, typename ValType>
class FileWriter
{
public:
  //print the info of N rows
  void write_block_csr(IdxType N, EIdType*ptr_start, IdxType*idx_start, ValType*val_start);
  void write_single_ptrval(EIdType*lastPtr);
  FileWriter(const std::string&name);
  FileWriter()=delete;
  FileWriter(const FileWriter&)=delete;
  void operator=(const FileWriter&)=delete;
  ~FileWriter();

private:
  FILE* m_fp_ptr;
  FILE* m_fp_idx;
  FILE* m_fp_val;
} ;

template class FileWriter<ull, float>;
template class FileWriter<IdxType, float>;

} //namespace IO



} // namespace CESpGEMM
