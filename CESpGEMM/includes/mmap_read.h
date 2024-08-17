#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <cctype>
#include <cstdlib>
#include "CSR.h"
#include <sstream>
#include <string>
#include <fstream>
#include <chrono>

namespace CESpGEMM{
namespace IO{

namespace{
struct MyFD{
  using string = std::string;
  int fd;
  char*data;
  size_t file_size, current_pos;
  MyFD(const string & name){
    fd = open(name.c_str(), O_RDONLY);
    if(fd<0){
      throw std::runtime_error(name+" Failed to open file");
    }
    struct stat st;
    if(fstat(fd, &st)<0){
      close(fd);
      throw std::runtime_error(name+" Failed to get file size.");
    } 
    file_size = st.st_size;
    data = static_cast<char*>(mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (data == MAP_FAILED) {
      close(fd);
      throw std::runtime_error("Failed to map file.");
    }
    current_pos = 0;
  }
  ~MyFD(){
    if (data) {
      printf("unmap data\n");
      munmap(data, file_size);
    }
    if (fd >= 0) {
      printf("close file!\n");
      close(fd);
    }
  }
  bool getLine(string &line){
    if (current_pos >= file_size) return false;
    size_t start_pos = current_pos;
    while (current_pos < file_size && data[current_pos] != '\n') {
      current_pos++;
    }
    line.assign(data + start_pos, current_pos - start_pos);
    if (current_pos < file_size && data[current_pos] == '\n') {
      current_pos++;
    }
    return true;
  }
} ;
}
template<typename eidType, typename valType>
class MMapReader {
private:
  using string = std::string;

public:
  MMapReader(const string & name): m_fd(std::make_unique<MyFD>(name)){}
  MMapReader(const MMapReader&) = delete;
  ~MMapReader() = default;

  void reopen(const string &name){
    m_fd = std::make_unique<MyFD>(name);
    // m_fd.reset(nullptr);
  }
  void clear(){
    m_fd.reset();
  }
  coo<eidType, valType> read_matrix_as_coo();
  csr<eidType, valType> preprocess(bool transpose, coo<eidType, valType>& contain);
  static std::tuple<size_t,size_t,size_t> readBanner(const std::string &name);

private:
  std::unique_ptr<MyFD> m_fd;
};

template<typename eidType, typename valType>
csr<eidType, valType> MMapReader<eidType, valType>::preprocess(bool transpose, coo<eidType, valType>& contain){
  using namespace std::chrono;
  printf("preprocess sizeof eidtype:%d\n",sizeof(eidType));
  if(contain.row==nullptr){
    contain = std::move(read_matrix_as_coo());
    printf("read file as coo: %d, %d, %lld, %p, %p, %p\n", contain.nr, contain.nc, contain.nnz, contain.row, contain.col, contain.val);
  }
  csr<eidType, valType> ret;
  printf("## preprocess sizeof eidtype:%d\n",sizeof(eidType));
  auto t0=steady_clock::now();
  ret.init_from_coo_cpu(contain, transpose);
  auto t1=steady_clock::now();
  printf("coo->csr used:%.3f\n", duration_cast<microseconds>(t1-t0).count()/1000.0);
  return ret;
}


template<typename eidType, typename valType>
coo<eidType, valType> MMapReader<eidType, valType>::read_matrix_as_coo() {
  std::string line_buffer;
  std::string matrix_market, mat_object, mat_format, mat_field, mat_sym;
  MyFD &fdref = *m_fd;
  
  {
    fdref.getLine(line_buffer);
    std::istringstream is(line_buffer);
    is>>matrix_market>>mat_object>>mat_format>>mat_field>>mat_sym;
  }
  std::cout << matrix_market << ' ' << mat_object << ' ' << mat_format << ' ' << mat_field << ' ' << mat_sym << "!\n";
  std::cout << "--------------------------------\n";
  std::cout << "matrix_market: " << matrix_market << "\n";
  std::cout << "matrix_object: " << mat_object << "\n";
  std::cout << "matrix_format: " << mat_format << "\n";
  std::cout << "matrix_field: " << mat_field << "\n";
  std::cout << "matrix_symmetry: " << mat_sym << "\n";
  std::cout << "--------------------------------\n";

  if (matrix_market != "%%MatrixMarket" || mat_object != "matrix" || mat_format != "coordinate") {
    throw std::runtime_error("Wrong format: should begin with %%MatrixMarket matrix coordinate.");
  }

  if (mat_field != "integer" && mat_field != "real" && mat_field != "pattern" && mat_field != "complex") {
    throw std::runtime_error("Matrix file should be: integer/real/pattern, but the input is: " + mat_field);
  }

  if (mat_sym != "symmetric" && mat_sym != "general" && mat_sym != "Hermitian") {
    throw std::runtime_error("Matrix symmetry should be symmetric/general/Hermitian, but the input is: " + mat_sym);
  }

  bool is_real = mat_field == "real";
  bool is_complex = mat_field == "complex";
  bool is_integer = mat_field == "integer";
  bool is_sym = mat_sym == "symmetric" || mat_sym == "Hermitian";

  while (line_buffer[0] == '%') {
    fdref.getLine(line_buffer);
  }

  eidType n, m, nnz;
  {
    char* end_ptr = nullptr;
    n = static_cast<eidType>(std::strtol(line_buffer.c_str(), &end_ptr, 10));
    m = static_cast<eidType>(std::strtol(end_ptr, &end_ptr, 10));
    nnz = static_cast<eidType>(std::strtol(end_ptr, &end_ptr, 10));
  }

  eidType real_nnz = nnz << is_sym;

  std::cout << "n=" << n << ", m=" << m << ", nnz=" << real_nnz << "\n";
  if (nnz == 0 || n == 0 || m == 0) throw std::runtime_error("Invalid matrix dimensions or nnz.");

  coo<eidType, valType> ret(n, m, real_nnz);
  eidType rows_read = 0;
  eidType idx = 0;

  while (rows_read < nnz) {
    fdref.getLine(line_buffer);
    if (line_buffer[0] == '%') continue;

    char* end_ptr = nullptr;
    IdxType r = static_cast<IdxType>(std::strtol(line_buffer.c_str(), &end_ptr, 10)) - 1;
    IdxType c = static_cast<IdxType>(std::strtol(end_ptr, &end_ptr, 10)) - 1;
    valType v = 1; // Default value for 'pattern' field

    if (is_real || is_integer || is_complex) {
      v = static_cast<valType>(std::strtod(end_ptr, &end_ptr));
    }

    ret.row[idx] = r;
    ret.col[idx] = c;
    ret.val[idx] = v;
    idx++;

    if (is_sym) {
      ret.row[idx] = c;
      ret.col[idx] = r;
      ret.val[idx] = v;
      idx++;
    }

    rows_read++;
  }

  std::cout << "Read file as COO: " << ret.nr << ", " << ret.nc << ", " << ret.nnz << "\n";
  return ret;
}


template<typename E, typename V>
std::tuple<size_t,size_t,size_t> MMapReader<E,V>::readBanner(const std::string &name){
  std::string line_buffer;
  std::ifstream fs(name);
  while(std::getline(fs, line_buffer)){
    if(line_buffer[0]=='%') continue;
    break;
  }
  std::stringstream ss;
  ss<<line_buffer;
  size_t row,col,nnz;
  ss>>row>>col>>nnz;
  return std::tuple(row, col, nnz);
}


}//IO
}//CESpGEMM

