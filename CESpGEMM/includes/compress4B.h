#pragma once
#include<cstdint>
#include<type_traits>
#include<vector>



enum CompCode4B{
  COMP4B_SUCCESS=0,
  COMP4B_EXCEED=1,
  COMP4B_ERROR=2
} ;

#define GetValue4B(a,i) a[3*(i)+0]
#define GetPreNz4B(a,i) a[3*(i)+1]
#define GetSrcPs4B(a,i) a[3*(i)+2]



struct Anchor4B{
  using u32 = uint32_t;
  u32 dstValue;
  u32 prefNZ;
  u32 srcPos;
  Anchor4B()=delete;
  Anchor4B(u32 dstV, u32 pref_nz, u32 srcP):dstValue(dstV), prefNZ(pref_nz), srcPos(srcP){};
} ;


template<typename T, int SL>
struct Compressor4B
{
  using u8 = unsigned char;
  using u16 = uint16_t;
  using u32 = uint32_t;
  using u64 = uint64_t;
  static constexpr int SegLen=SL;
  
  u32 nbytes_control;
  u32 num_values;
  u32 bytes_of_data;
  u32 num_segs;

  u8 * control;
  u8 * data;
  // Anchor * anchor;
  u32 * anchor_data;

  Compressor4B(const Compressor4B&)=delete;
  void operator=(const Compressor4B&)=delete;
  Compressor4B():data(nullptr), control(nullptr), anchor_data(nullptr){
    static_assert(SegLen <= 1024);
  }
  ~Compressor4B();

  // enum CompCode compress(int nElems, const T *src, int nz_threshold, int maxNsegs);
  enum CompCode4B compress_with_zero_per_seg(int nElems, const T *src, int nz_threshold, int maxNsegs_perseg);
	size_t get_compressed_size() const{
		size_t ans = nbytes_control;
		ans += bytes_of_data;
		ans += 3*sizeof(int)*(num_segs+1);
		return ans;
	}
} ;

template struct Compressor4B<unsigned, 512>;
template struct Compressor4B<int, 512>;

using comp_4b_type = Compressor4B<unsigned, 512>;
