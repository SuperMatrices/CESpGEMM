#pragma once
#include<cstdint>
#include<cstdio>
#include<vector>
#include<cassert>
#include"CSR.h"

enum CompCode{
  COMP_SUCCESS=0,
  COMP_EXCEED=1,
  COMP_ERROR=2
} ;

#define GetValue(a,i) a[3*(i)+0]
#define GetPreNz(a,i) a[3*(i)+1]
#define GetSrcPs(a,i) a[3*(i)+2]

struct Anchor{
  using u32 = uint32_t;
  u32 dstValue;
  u32 prefNZ;
  u32 srcPos;
  Anchor()=delete;
  Anchor(u32 dstV, u32 pref_nz, u32 srcP):dstValue(dstV), prefNZ(pref_nz), srcPos(srcP){};
} ;



// struct ZAnchor{
//   using u8 = unsigned char;
//   using u32 = uint32_t;
//   u32 pref_sum;
// } ;


template<typename T>
struct Compressor
{
  using u8 = unsigned char;
  using u16 = uint16_t;
  using u32 = uint32_t;
  using u64 = uint64_t;
  
  const u32 SegLen;
  u32 nbytes_control;
  u32 num_values;
  u32 bytes_of_data;
  u32 num_segs;

  u8 * control;
  u8 * data;
  // Anchor * anchor;
  u32 * anchor_data;

  // Compressor(const Compressor&)=delete;
  // void operator=(const Compressor&)=delete;
  Compressor(u32 seglen):data(nullptr), control(nullptr), anchor_data(nullptr), SegLen(seglen){}
  Compressor()=delete;
  ~Compressor();

  // enum CompCode compress(int nElems, const T *src, int nz_threshold, int maxNsegs);
  enum CompCode compress_with_zero_per_seg(int nElems, const T *src, int nz_threshold, int maxNsegs_perseg);

  
} ;

template struct Compressor<int>;
template struct Compressor<unsigned>;

struct CompInfo{
  int seglen;
  int min_zero_num;
  int max_zero_seg;
  CompInfo(int seglen, int min_zero_num, int max_zero_seg):seglen(seglen), min_zero_num(min_zero_num),max_zero_seg(max_zero_seg) { }
} ;

namespace CESpGEMM{
  using compress_t = Compressor<IdxType>;
}


