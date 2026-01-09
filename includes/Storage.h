#pragma once

#include"CSR.h"
#include<vector>
#include<atomic>
#include"compressor.h"

namespace CESpGEMM
{

/**
 * @brief Global singleton storage for sparse matrix multiplication (SpGEMM) data
 * 
 * This template class implements a singleton pattern to manage all the data structures
 * required for sparse matrix multiplication operations. It stores matrices A and B_T (transposed B),
 * manages compressed representations, and tracks computational metrics like FLOPs (floating point operations).
 * 
 * @tparam SrcEType Source element type for matrix indices (e.g., IdxType for 32-bit, ull for 64-bit)
 * @tparam EIdType Element ID type used internally for indexing
 * @tparam ValType Value type for matrix elements (typically float or double)
 */
template<typename SrcEType, typename EIdType, typename ValType>
class GlobalStorage 
{
public:
  /// Number of grids used for merging results on GPU (affects parallelism and memory usage)
  static constexpr int nGridsMerge = 256;
  
  /// Type alias for source element type (indices from input matrices)
  using srcEtype = SrcEType;
  
  /// Type alias for element ID type (used internally for indexing)
  using eidType = EIdType;
  
  /// Type alias for value type (matrix element values)
  using valType = ValType;

  /// Size of memory pool for intermediate results (in bytes)
  size_t pool_size;
  
  /// Number of worker threads for parallel processing
  int num_workers;
  
  // ValType* cpu_dense;  // [Deprecated] Dense CPU storage, no longer used
  
  /// Block size for matrix A (rows per block)
  IdxType blockSizeA;
  
  /// Block size for matrix B (columns per block)
  IdxType blockSizeB;
  
  /// Total number of blocks along matrix A's rows
  IdxType numBlocksA;
  
  /// Total number of blocks along matrix B's columns
  IdxType numBlocksB;
  
  /// Compression configuration parameters (segment length, zero thresholds, etc.)
  CompInfo cmpInfo;

  /// Shared pointer to matrix A in CSR format (Compressed Sparse Row)
  shared_ptr<const csr<SrcEType, ValType> > csrA;
  
  /// Shared pointer to transposed matrix B in CSC format (Compressed Sparse Column)
  shared_ptr<const csr<SrcEType, ValType> > csrB_T;
  
  /// 64-bit index array for matrix A (used when SrcEType is 64-bit)
  /// Allocated only when sizeof(SrcEType) == 8 to avoid memory overhead
  ull *csra_idx_64;
  
  /// 64-bit index array for matrix B (used when SrcEType is 64-bit)
  /// Allocated only when sizeof(SrcEType) == 8 to avoid memory overhead
  ull *cscb_idx_64;
  
  // std::vector<csr<EIdType, ValType>> vcsrB;  // [Deprecated] Vector CSR format for B
  
  /// Vector of raw CSR structures for matrix B blocks
  /// Each element contains only idx and val arrays (no ptr array)
  /// Used for efficient block-wise processing
  std::vector< std::unique_ptr< raw_csr< ValType > > > vcsrb_raw;
  
  /// Vector of compressors, one for each block of matrix B
  /// Each compressor handles the compressed representation of a B block
  std::vector<compress_t> v_comp_ptr;
  
  /// Estimated FLOPs (floating point operations) for each block
  /// Used for workload balancing and task scheduling
  std::vector<ull> block_flops;
  
  /// Threshold FLOP count for determining if a task should run on GPU
  /// Tasks with FLOPs below this threshold may be assigned to CPU
  ull gpu_flop_thresh;
  
  /// Maximum FLOPs among all row-column block pairs
  /// Used for resource allocation and scheduling decisions
  ull max_rcblock_flop;
  
  /// Maximum GPU memory available for computation (in bytes)
  size_t max_gpu_bytes;
  
  /// Maximum allowed FLOPs considering GPU memory constraints
  /// Computed as: (max_gpu_bytes - overhead) / (sizeof(IdxType) + sizeof(ValType))
  size_t max_allowed_flop;
  
  /// Flag to enable/disable file writing for debugging or logging
  bool enable_write;

  /// Type alias for convenience (vector template)
  template<typename T>
  using vec = std::vector<T>;

  /**
   * @brief Get the singleton instance of GlobalStorage
   * @return Pointer to the singleton instance, or nullptr if not initialized
   */
  static GlobalStorage *Instance();
  
  /**
   * @brief Initialize or reinitialize the singleton GlobalStorage instance
   * 
   * This method creates the singleton instance with the specified parameters.
   * If an instance already exists, it will be deleted and replaced.
   * 
   * @param blockSizeA Number of rows per block for matrix A
   * @param blockSizeB Number of columns per block for matrix B
   * @param numBlocksA Total number of blocks along A's rows
   * @param numBlocksB Total number of blocks along B's columns
   * @param poolSize Size of memory pool for intermediate results (bytes)
   * @param num_workers Number of worker threads
   * @param a Shared pointer to matrix A in CSR format
   * @param bT Shared pointer to transposed matrix B in CSC format
   * @param block_flops Vector of estimated FLOPs for each block (moved)
   * @param gpu_flop_thresh FLOP threshold for GPU task assignment
   * @param max_gpu_bytes Maximum GPU memory available (bytes)
   * @param file_write Enable file writing for debugging
   * @param comp_info Compression configuration parameters
   */
  static void Init(IdxType blockSizeA, IdxType blockSizeB, IdxType numBlocksA, IdxType numBlocksB, size_t poolSize, int num_workers, shared_ptr<csr<SrcEType, ValType>> a, shared_ptr<csc<SrcEType,ValType>> bT, std::vector<ull> && block_flops, ull gpu_flop_thresh, size_t max_gpu_bytes, bool file_write, CompInfo comp_info);

private:
  /// Static pointer to the singleton instance
  static GlobalStorage* gs;
  
  /// Default constructor is deleted (singleton pattern)
  GlobalStorage()=delete;
  
  /**
   * @brief Private constructor for GlobalStorage
   * 
   * Initializes all member variables and performs preprocessing:
   * - Converts matrix B blocks to raw CSR format
   * - Compresses each B block using the configured compressor
   * - Computes FLOP estimates for workload balancing
   * - Allocates 64-bit index arrays if needed
   * 
   * @param blockSizeA Number of rows per block for matrix A
   * @param blockSizeB Number of columns per block for matrix B
   * @param numBlocksA Total number of blocks along A's rows
   * @param numBlocksB Total number of blocks along B's columns
   * @param poolSize Size of memory pool for intermediate results (bytes)
   * @param num_workers Number of worker threads
   * @param a Shared pointer to matrix A in CSR format
   * @param bT Shared pointer to transposed matrix B in CSC format
   * @param block_flops Vector of estimated FLOPs for each block (moved)
   * @param gpu_flop_thresh FLOP threshold for GPU task assignment
   * @param max_gpu_bytes Maximum GPU memory available (bytes)
   * @param file_write Enable file writing for debugging
   * @param comp_info Compression configuration parameters
   */
  GlobalStorage(IdxType blockSizeA, IdxType blockSizeB, IdxType numBlocksA, IdxType numBlocksB, size_t poolSize, int num_workers, shared_ptr<csr<SrcEType, ValType>> a, shared_ptr<csc<SrcEType,ValType>> bT, std::vector<ull> && block_flops, ull gpu_flop_thresh, size_t max_gpu_bytes, bool file_write, CompInfo comp_info);
  
  /**
   * @brief Destructor - cleans up allocated resources
   * 
   * Frees 64-bit index arrays if they were allocated
   * (only when sizeof(SrcEType) == 8)
   */
  ~GlobalStorage();
} ;



/// Explicit template instantiation for 64-bit indices (ull), 32-bit IDs, float values
template class GlobalStorage<ull, IdxType, float>;

/// Explicit template instantiation for 32-bit indices (IdxType), 32-bit IDs, float values
template class GlobalStorage<IdxType, IdxType, float>;

/// Type alias for default GlobalStorage configuration (32-bit indices, float values)
/// Used for matrices with up to ~4 billion non-zero elements
using default_gs_t = GlobalStorage<IdxType, IdxType, float>;

/// Type alias for large GlobalStorage configuration (64-bit indices, float values)
/// Used for matrices with more than ~4 billion non-zero elements
using large_gs_t = GlobalStorage<ull, IdxType, float>;


} // namespace CESpGEMM
