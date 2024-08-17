#pragma once
#include"CSR.h"
#include<mkl.h>
#include<mkl_spblas.h>
#include<memory>
#include<type_traits>

namespace CESpGEMM
{
inline void chk_mkl_err(sparse_status_t err, const char*func, const char* file, int line){
  if(err!=0){
    printf("error code = %d\n", err);
    printf(
     "SPARSE_STATUS_SUCCESS           = 0,    /* the operation was successful */\n\
      SPARSE_STATUS_NOT_INITIALIZED   = 1,    /* empty handle or matrix arrays */\n\
      SPARSE_STATUS_ALLOC_FAILED      = 2,    /* internal error: memory allocation failed */\n\
      SPARSE_STATUS_INVALID_VALUE     = 3,    /* invalid input value */\n\
      SPARSE_STATUS_EXECUTION_FAILED  = 4,    /* e.g. 0-diagonal element for triangular solver, etc. */\n\
      SPARSE_STATUS_INTERNAL_ERROR    = 5,    /* internal error */\n\
      SPARSE_STATUS_NOT_SUPPORTED     = 6     /* e.g. operation for double precision doesn't support other types */\n");
    throw;
  }
}

#define CHK_MKL(val) ::CESpGEMM::chk_mkl_err((val), #val, __FILE__, __LINE__)

template<bool LONG>
struct MKL_Func_Factory{
  template<typename... Args>
  static void create_csr(Args&&...args){
    CHK_MKL( mkl_sparse_s_create_csr(std::forward<Args>(args)...) );
  }
  template<typename... Args>
  static void convert_csr(Args&&...args){
    CHK_MKL( mkl_sparse_convert_csr(std::forward<Args>(args)...) );
  }
  template<typename... Args>
  static void create_csc(Args&&...args){
    CHK_MKL( mkl_sparse_s_create_csc(std::forward<Args>(args)...) );
  }
  template<typename... Args>
  static void spgemm(Args&&...args){
    CHK_MKL( mkl_sparse_spmm(std::forward<Args>(args)...) );
  }
  template<typename... Args>
  static void export_csr(Args&&...args){
    CHK_MKL( mkl_sparse_s_export_csr(std::forward<Args>(args)...) );
  }
  template<typename... Args>
  static void destroy_csr(Args&&...args){
    CHK_MKL( mkl_sparse_destroy(std::forward<Args>(args)...) );
  }
} ;



template<>
struct MKL_Func_Factory<true>
{
  template<typename... Args>
  static void create_csr(Args&&...args){
    CHK_MKL( mkl_sparse_s_create_csr_64(std::forward<Args>(args)...) );
  }
  template<typename... Args>
  static void convert_csr(Args&&...args){
    CHK_MKL( mkl_sparse_convert_csr_64(std::forward<Args>(args)...) );
  }
  template<typename... Args>
  static void create_csc(Args&&...args){
    CHK_MKL( mkl_sparse_s_create_csc_64(std::forward<Args>(args)...) );
  }
  template<typename... Args>
  static void spgemm(Args&&...args){
    CHK_MKL( mkl_sparse_spmm_64(std::forward<Args>(args)...) );
  }
  template<typename... Args>
  static void export_csr(Args&&...args){
    CHK_MKL( mkl_sparse_s_export_csr_64(std::forward<Args>(args)...) );
  }
  template<typename... Args>
  static void destroy_csr(Args&&...args){
    CHK_MKL( mkl_sparse_destroy_64(std::forward<Args>(args)...) );
  }
} ;

template<typename EIdType>
struct spMatDeleter{
  using funfact= MKL_Func_Factory<sizeof(EIdType)==8>;
  void operator()(sparse_matrix *psp){
    sparse_matrix_t sp = psp;
    funfact::destroy_csr(sp);
  }
} ;

template<typename EIdType, typename ValType>
struct MKLSpMat{
  using deleter_t = spMatDeleter<EIdType>;
  using mtype = std::unique_ptr<sparse_matrix, deleter_t>;
  using IntType = std::conditional_t< sizeof(EIdType)==4 , MKL_INT, MKL_INT64 >;
  using func_util = MKL_Func_Factory<sizeof(EIdType)==8>;

  static mtype CreateUniqueMatCsr(EIdType nr, EIdType nc, EIdType*ptr, EIdType*idx, ValType*val){
    // static_assert(sizeof(EIdType)==4);
    static_assert(std::is_same<ValType, float>::value);
    sparse_matrix_t spm;
    // mkl_sparse_s_create_csr
    func_util::create_csr(&spm, SPARSE_INDEX_BASE_ZERO, nr, nc, 
      reinterpret_cast<IntType*>(ptr),
      reinterpret_cast<IntType*>(ptr+1),
      reinterpret_cast<IntType*>(idx),
      val
    );
    return mtype{spm, deleter_t{}};    
  }
  
  static void ConvertCsr(mtype &csc_mat){
    sparse_matrix_t csc_p = csc_mat.release(), csr_p;
    func_util::convert_csr(csc_p, SPARSE_OPERATION_NON_TRANSPOSE, &csr_p);
    // csc_mat.release();
    csc_mat.reset(csr_p);
  }
  
  static mtype CreateUniqueMatCsc(EIdType nr, EIdType nc, EIdType*ptr, EIdType*idx, ValType*val){
    sparse_matrix_t spm;
    func_util::create_csc(&spm, SPARSE_INDEX_BASE_ZERO, nr, nc, 
      reinterpret_cast<IntType*>(ptr),
      reinterpret_cast<IntType*>(ptr+1),
      reinterpret_cast<IntType*>(idx),
      val
    );
    return mtype{spm, deleter_t{}};    
  }
  static mtype Mult(const mtype &a ,const mtype&b){
    sparse_matrix_t spc;
    func_util::spgemm(SPARSE_OPERATION_NON_TRANSPOSE, a.get(), b.get(), &spc);
    return mtype{spc, deleter_t{}};
  } 
} ;
  
} // namespace CESpGEMM
