#include"FileIO.h"
#include"profiler.h"
#include<sstream>
#include<fstream>

namespace CESpGEMM
{ 

namespace IO{

// FileReader: Reads sparse matrices from Matrix Market format files
template<typename e, typename v>
char FileReader<e,v>::gchar(){
  MyFP &F=*_myfp.get();
  if(F.check_end()) return EOF;
  return *(F._st++);
}

// Read a single line from the file
template<typename e, typename v>
inline void FileReader<e,v>::getLine(std::string&s){
  s="";
  char c=gchar();
  if(c==EOF)return;
  while(c!='\n'){
    s+=c;
    c=gchar();
  }
}

// =====
template<typename E, typename V>
std::tuple<size_t,size_t,size_t> FileReader<E,V>::readBanner(const std::string &name){
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

// Read matrix banner (dimensions and nnz) from Matrix Market file
template<typename eidType,typename valType>
coo<eidType,valType> FileReader<eidType,valType>::read_matrix_as_coo(){
  std::string line_buffer;
  std::string matrix_market, mat_object, mat_format, mat_field, mat_sym;
  {
    getLine(line_buffer);
    std::istringstream is(line_buffer);
    is>>matrix_market>>mat_object>>mat_format>>mat_field>>mat_sym;
  }
  
  std::cout<<matrix_market<<' '<<mat_object<<' '<<mat_format<<' '<<mat_field<<' '<<mat_sym<<"!\n";
  printf("--------------------------------\n");
  printf("matrix_market: %s\n", matrix_market.c_str());
  printf("matrix_object: %s\n", mat_object.c_str());
  printf("matrix_format: %s\n", mat_format.c_str());
  printf("matrix_field: %s\n", mat_field.c_str());
  printf("matrix_symmetry: %s\n", mat_sym.c_str());
  printf("--------------------------------\n");
  // Validate Matrix Market format header
  if(matrix_market != "%%MatrixMarket" || mat_object!="matrix" || mat_format!="coordinate"){
    std::cout<<"Wrong format: should begin with %%MatrixMarket matrix coordinate\n";
    throw std::exception();
  }
  // Validate matrix field type (integer/real/pattern/complex)
  if(mat_field != "integer" && mat_field != "real" && mat_field != "pattern" && mat_field!="complex"){
    std::cout<<"matrix file should be: integer/real/pattern, but the input is: "<<mat_field<<'\n';
    throw std::exception();
  }
  // Validate matrix symmetry type
  if(mat_sym != "general" && mat_sym != "symmetric" && mat_sym!="Hermitian"){
    std::cout<<"matrix symmetry should be symmetric/general, but the input is: "<<mat_sym<<"\n";
    throw std::exception();
  }
  // Parse matrix properties from header
  bool is_real = mat_field == "real";
  bool is_complex = mat_field == "complex";
  bool is_integer = mat_field == "integer";
  bool is_sym = mat_sym == "symmetric"|| mat_sym=="Hermitian";
  while(line_buffer[0]=='%'){
    getLine(line_buffer);
  };
  IdxType n,m;
  eidType nnz;
  
  {
    std::istringstream is(line_buffer);
    is>>n>>m>>nnz;
  }
  // For symmetric matrices, nnz will double (store both i,j and j,i)
  eidType real_nnz = nnz << is_sym;

  printf("n=%d, m=%d, nnz=%lld\n", n,m, (long long)real_nnz);
  if(nnz==0||n==0||m==0) throw std::exception();

  coo<eidType, valType> ret(n, m, real_nnz);
  printf("ret %p %p %p\n", ret.row, ret.col, ret.val);
  eidType rows_read = 0;
  eidType idx=0;
  while(rows_read<nnz){
    getLine(line_buffer);
    if(line_buffer[0]=='%') continue;
    std::stringstream is(line_buffer);
    IdxType r,c;
    valType v;
    is>>r>>c;
    if(is_real||is_integer||is_complex){
      is>>v;
    }
    else{
      v = 1;
    }
    --r, --c;
    ret.row[idx] = r;
    ret.col[idx] = c;
    ret.val[idx] = v;
    idx++;
    if(is_sym){
      // Add symmetric entry (j,i) for symmetric matrices
      ret.row[idx] = c;
      ret.col[idx] = r;
      ret.val[idx] = v;
      idx++;
    }
    rows_read ++;
  }
  return ret;
  // return std::move(ret);
}

// Preprocess matrix: read from file, convert to CSR format
// Optionally transpose the matrix during conversion
template<typename EIdType, typename ValType>
csr<EIdType, ValType> FileReader<EIdType, ValType>::preprocess(bool transpose, coo<EIdType, ValType>& contain){
  if(contain.row == nullptr){
    contain = std::move(read_matrix_as_coo());
  }
  csr<EIdType, ValType> ret;
  ret.init_from_coo_cpu(contain, transpose);
  return ret;
}



} // namespace IO
} // namespace CESpGEMM
