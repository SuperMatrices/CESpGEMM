#include"compress4B.h"
#include<cassert>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#define TemplateLine template<typename T,int SL>


namespace
{

template<typename T>
T* copy_vec(const std::vector<T>&v){
  T*ret = (T*) malloc(sizeof(T)* v.size());
  memcpy(ret, v.data(), sizeof(T)*v.size());
  return ret;
}



  
} // namespace 


template<typename T, int cap, typename = std::enable_if_t<cap<=2> >
struct Queue
{
  static constexpr int N = cap;
  T q[N+2];
  int len;
  Queue(){
    len = 0;
  }
  void push(T x){
    assert(len<=N);
    int i=len-1;
    #pragma unroll
    for(;i>0;i--){
      if(q[i]>=x) break;
      q[i+1] = std::move(q[i]);
    }
    q[i+1] = x;
    len++;
  }
  void pop(){
    len --;
  }
  T back() const{
    return q[len-1];
  }
  int size() const{
    return len;
  }
  void clear(){
    len = 0;
  }
} ;

struct Seg{
  using u32=uint32_t;
  u32 l,r;
  bool is_zero;
  Seg(u32 L, u32 R, bool zero): l(L), r(R), is_zero(zero){}
  u32 getLength(){
    return r-l+1;
  }
} ;

TemplateLine
Compressor4B<T,SL>::~Compressor4B(){
  if(this->anchor_data) free(this->anchor_data);
  if(this->data) free(this->data);
  if(this->control) free(this->control);
}

TemplateLine
enum CompCode4B Compressor4B<T,SL>::compress_with_zero_per_seg(int nElems, const T *src, int nz_threshold, int maxNsegs_perseg){
  using std::vector, std::pair;
  using Pq = Queue<std::pair<int,int>, 2>;
  assert(maxNsegs_perseg <= 2);
  Pq q_zero_segs;
  vector<Seg>segs;
  
  vector<u8> tgt_data_bytes;
  vector<u8> tgt_control;
  vector<Anchor4B> tgt_anchors;
  u32 sum_values = 0, pref_nzeros=0;

  auto Consume=[this, src, &segs, &q_zero_segs, &tgt_data_bytes, &tgt_control, &tgt_anchors, &sum_values, &pref_nzeros] (u32 &total_length) -> enum CompCode4B {
    tgt_anchors.emplace_back(sum_values, pref_nzeros, tgt_data_bytes.size());
    int nsegs = segs.size();
    vector<u32> relPos(nsegs+1);
    vector<Seg> zero_segs_new;
    int idx = 0, lastId = 0;
    u32 sumval_in_seg = 0;
    u32 nzeros_in_seg = 0;
    u32 rem = 0;
    decltype(segs.begin()) it = segs.begin();
    vector<pair<int,u32>>record_zero_head;
    u32 deleted_length = 0;
    for(idx=0;it!=segs.end();it++, idx++){
      u32 l = it->l;
      u32 r = it->r;
      bool is_zero = it->is_zero;
      u32 this_len = (!is_zero)*(r-l+1);
      deleted_length += this_len;
      u32 pos_end = relPos[idx]+this_len;
      relPos[idx+1] = pos_end;
      if(pos_end >= SegLen){
        lastId = idx;
        rem = pos_end - SegLen;
        deleted_length -= rem;
        break;
      }
      if(is_zero){
        record_zero_head.push_back({relPos[idx], r-l+1});
      }
    }
    if(it == segs.end()) lastId = segs.size()-1;
    q_zero_segs.clear();
    if(it!=segs.end()) {
      ++idx;
      ++it;
    }
    for(;it!=segs.end();++it, ++idx) if(it->is_zero){
      u32 l = it->l;
      u32 r = it->r;
      if(!rem){
        q_zero_segs.push({r-l+1, idx - lastId -1});
        // printf("qpush id=%d, len=%d\n", idx-lastId-1, r-l+1);
      }
      else{
        q_zero_segs.push({r-l+1, idx-lastId});
        // printf("qpush id=%d, len=%d\n", idx-lastId, r-l+1);
      }
    }
    tgt_data_bytes.push_back(record_zero_head.size());
    for(auto [rel_pos, length] : record_zero_head){
      tgt_data_bytes.push_back(rel_pos & 255);
      tgt_data_bytes.push_back(rel_pos / 256);
      tgt_data_bytes.push_back(length & 255);
      tgt_data_bytes.push_back((length>>8) & 255);
      tgt_data_bytes.push_back((length>>16) & 255);
      tgt_data_bytes.push_back((length>>24) & 255);
      nzeros_in_seg += length;
    }
    const u32 previous_num_values = (tgt_anchors.size() - 1)*SegLen;
    const u32 current_num_values = previous_num_values + relPos[lastId+1];

    if(current_num_values /4 >= tgt_control.size()){
      tgt_control.resize(current_num_values/4+1);
    }
    for(idx=0,it=segs.begin();idx<=lastId;idx++, it++){
      if(it->is_zero) continue;
      u32 rel_pos = relPos[idx];
      u32 l = it->l, r=it->r;
      if(idx==lastId){
        r -= rem;
      }
      for(u32 j=l;j<=r;j++){
        u8 nb = -1;
        u32 val = static_cast<u32>(src[j]);
        u32 val_back = val;
        // assert(val<12288);
        // if(val>=256*256) printf("val=%d,",val);
        sumval_in_seg += val;
        do{
          tgt_data_bytes.push_back(val & 255);
          val /=256;
          nb++;
        }while(val);
        u32 elemId = previous_num_values + rel_pos + j-l;
        if(val_back>10000){
          // printf("elemId=%d, val=%d\n", elemId, val_back);
          if(nb>1) printf("elemId=%d, nb=%d, val=%lld\n", elemId, nb, val_back);
        }
        tgt_control[elemId /4] |= (nb << ((elemId & 3)*2));
      }
    }
    if(rem){
      segs.erase(segs.begin(), segs.begin() + lastId);
      segs[0].l = segs[0].r - rem+1;
    }else{
      segs.erase(segs.begin(), segs.begin() + lastId + 1);
    }
    total_length -= deleted_length;
    sum_values += sumval_in_seg ;
    pref_nzeros += nzeros_in_seg;
    return CompCode4B::COMP4B_SUCCESS;
  } ;

  u32 total_length = 0;
  u32 nnz = nElems;
  for(u32 i=0;i<nElems; ++i /* in each branch the end of a contiguous segment is reached*/){
    if(src[i]){
      u32 start = i;
      while(i+1<nElems && src[i+1]) ++i;
      // printf("push (%d,%d)\n", start, i);
      if(!segs.size() || segs.back().is_zero ){
        segs.push_back(Seg{start, i, false});
        total_length += i-start+1;
        // printf("push back: %d,%d,%d\n", 0, i, false);
      }else{
        total_length += i-segs.back().r;
        segs.back().r = i;
      }
      // printf("total_length now=%d\n", total_length);
      // for(auto s:segs){
      //   printf("slen=%d\n", s.getLength());
      // }
      while(total_length >= SegLen){
        CompCode4B gather_result = Consume(total_length);
        if(gather_result != CompCode4B::COMP4B_SUCCESS){
          return gather_result;
        }
      }
    }
    else{
      u32 start = i;
      while(i+1<nElems && !src[i+1]) ++i;
      if(i-start+1>=nz_threshold){
        segs.emplace_back(start, i, true);
        nnz -= i-start+1;
        q_zero_segs.push({i-start+1, segs.size()-1});
        if(q_zero_segs.size() > maxNsegs_perseg){
          auto [len,id]=q_zero_segs.back();
          segs[id].is_zero = false;
          nnz += len;
          total_length += len;
          q_zero_segs.pop();
        }
      }
      else{
        if(!segs.size()||segs.back().is_zero){
          segs.emplace_back(start, i, false);
          total_length += i-start+1;
        }
        else{
          total_length += i-segs.back().r;
          segs.back().r = i;
        }
      }
    }
  }
  while(total_length){
    CompCode4B gather_res = Consume(total_length);
    if(gather_res != CompCode4B::COMP4B_SUCCESS){
      return gather_res;
    }
  }
  tgt_anchors.emplace_back(sum_values, pref_nzeros, tgt_data_bytes.size());
  this->nbytes_control = tgt_control.size();
  this->control = copy_vec(tgt_control);
  this->bytes_of_data = tgt_data_bytes.size();
  this->data = copy_vec(tgt_data_bytes);
  // this->anchor = copy_vec(tgt_anchors);
  this->anchor_data = (u32*) malloc( tgt_anchors.size() * 3 * sizeof(u32));
  for(int i=0;i<(int)tgt_anchors.size();i++){
    auto [val, nz, pos] = tgt_anchors[i];
    GetValue4B(anchor_data, i) = val;
    GetPreNz4B(anchor_data, i) = nz;
    GetSrcPs4B(anchor_data, i) = pos;
  }

  this->num_segs = tgt_anchors.size()-1;
  this->num_values = nnz;

  return CompCode4B::COMP4B_SUCCESS;
}

#undef TemplateLine