#include"compressor.h"
#include<list>
#include<vector>
#include<cstring>
#include<algorithm>
#include<functional>
#include<cassert>
#include<queue>
#include<iostream>



namespace{

struct ZeroSeg{
  uint32_t len;
  uint32_t start;
  ZeroSeg(uint32_t l,uint32_t start_pos): len(len), start(start){} 
  bool operator<(const ZeroSeg& zs)const{
    if(len!=zs.len) len > zs.len;
    return start > zs.start;
  }
} ;//将长度>nz_threshold的0段，首先默认记为加入该段的zerohead中，如果数量超过maxNsegs,每次把最短的段pop出来

template<typename T>
T* copy_vec(const std::vector<T> &v){
  T* ret = (T*) malloc(sizeof(T)* v.size());
  memcpy(ret, v.data(), sizeof(T)*v.size());
  return ret;
}

}

struct Seg{
  using u32=uint32_t;
  u32 l,r;
  bool is_zero;
  Seg(u32 L, u32 R, bool zero): l(L), r(R), is_zero(zero){}
  u32 getLength(){
    return r-l+1;
  }
} ;

template<typename T, int SL>
Compressor<T,SL>::~Compressor(){
  if(this->anchor_data) free(this->anchor_data);
  if(this->data) free(this->data);
  if(this->control) free(this->control);
}

template<typename T, int SL>
enum CompCode Compressor<T, SL>::compress_with_zero_per_seg(int nElems, const T*src, int nz_threshold, int maxNsegs_perseg){
  using std::vector, std::pair, std::list, std::priority_queue;
  assert(maxNsegs_perseg <= 2);

  vector<Seg> segments;
  priority_queue< pair<u32,u32>, vector<pair<u32,u32>>, std::greater<pair<u32,u32>> > q_zero_segs;
  
  vector<u8> tgt_data_bytes;
  vector<u8> tgt_control;
  vector<Anchor> tgt_anchors;
  u32 sum_values = 0, pref_nzeros=0;
  
  auto printqueue=[](priority_queue< pair<u32,u32>, vector<pair<u32,u32>>, std::greater<pair<u32,u32>> > q){
    while(!q.empty()){
      auto k = q.top();
      printf("(%d,%d),",k.first, k.second);
      q.pop();
    }
    printf("\n");
  } ;

  auto Consume = [this, src, &segments, &q_zero_segs, &tgt_data_bytes, &tgt_control, &tgt_anchors, &sum_values, &pref_nzeros, printqueue] (u32 &total_length) -> enum CompCode {
    // printf("Consume!, numSegs=%d, total_length=%d, length0=%d, sum_values=%d, prefnz=%d, first=%d, last=%d\n", segments.size(), total_length, segments[0].r-segments[0].l+1, sum_values, pref_nzeros, segments[0].l, segments.back().r);
    // u32 count_total = 0;
    // for(auto an: segments){
    //   count_total += (an.r-an.l+1) * (!an.is_zero);
    // }
    // printqueue(q_zero_segs);
    // printf("count_total=%d, total_length=%d!\n", count_total, total_length);
    // if(count_total!=total_length) printf("beforedifferent\n");
    tgt_anchors.emplace_back(sum_values, pref_nzeros, tgt_data_bytes.size());

    
    int nsegs = segments.size();
    vector<u32> relPos(nsegs+1);
    vector<Seg> zero_segs_new;
    
    int idx=0, lastId = 0;
    u32 sumval_in_seg = 0;
    u32 nzeros_in_seg = 0;
    u32 rem = 0;
    auto it = segments.begin();
    vector<pair<int, u32> > record_zero_head; 
    u32 deleted_length = 0;

    for(idx=0;it!=segments.end();it++, idx++){
      u32 l = it->l;
      u32 r = it->r;
      bool is_zero = it->is_zero;
      u32 this_len = (!is_zero) * (r-l+1);
      deleted_length += this_len;
      u32 pos_end = relPos[idx] + this_len;
      relPos[idx+1] = pos_end;
      if(pos_end >= SegLen) {
        lastId = idx;
        rem = pos_end - SegLen;
        deleted_length -= rem;
        break;
      }
      if(is_zero){
        record_zero_head.push_back({relPos[idx], r-l+1});
      }
    }
    if(it == segments.end()) lastId = segments.size()-1;
    // printf("Consume rem=%d, lastId=%d, last of last=%d\n", rem, lastId, segments[lastId].r);
    while(!q_zero_segs.empty()) q_zero_segs.pop();
    if(it!=segments.end()){
      ++idx;
      ++it;
    }
    for(;it!=segments.end();it++, idx++) if(it->is_zero){
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
    for(auto [rel_pos, length]: record_zero_head) {
      // printf("zero relpos=%d, length=%d\n", rel_pos, length);
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
    if(current_num_values/8 >= tgt_control.size()){
      tgt_control.resize(current_num_values/8+1);
    }
    // printf("consume getting elems\n");

    for(idx=0, it = segments.begin(); idx<=lastId ; idx++, it++){
      if(it->is_zero) continue;
      u32 rel_pos = relPos[idx];
      u32 l = it->l, r=it->r;
      if(idx==lastId){
        r -= rem;
      }
      for(u32 j=l;j<=r;j++){
        u32 val = static_cast<u32>(src[j]);
        // printf("j=%d,val=%d\n",j,val);
        if(val>=65536) return CompCode::COMP_EXCEED;
        u8 more_bytes = (val>=256);
        u32 elemId = previous_num_values + rel_pos + j - l;
        tgt_data_bytes.push_back(val&255);
        if(more_bytes) tgt_data_bytes.push_back(val/256);
        tgt_control[elemId/8] |= more_bytes<<(elemId & 7);
        // if(more_bytes) printf("val:%d, morebytes%d, \n", val, more_bytes);
        if(tgt_control.size() <= elemId/8){
          fprintf(stderr, "tgt_control out : eid=%d, size=%d, prefnv=%d, endnv=%d\n", elemId, tgt_control.size(), previous_num_values, current_num_values);
        }
        sumval_in_seg += val;
      }
    } 
    
    if(rem){
      segments.erase(segments.begin(), segments.begin()+lastId);
      segments[0].l = segments[0].r - rem + 1;
    }
    else{
      segments.erase(segments.begin(), segments.begin()+lastId+1);
    }
    total_length -= deleted_length;
    // printf("after Consume: deleted length=%d, nsegs=%d\n", deleted_length, segments.size());
    // count_total = 0;
    // for(auto au : segments){
    //   count_total += (au.r-au.l+1)*(!au.is_zero);
    //   // printf("(%d,%d,%d)\n", au.l,au.r, au.is_zero);
    // }
    // printf("after consume: countlen=%d, totlen=%d, qsize=%d\n", count_total, total_length, q_zero_segs.size());
    // if(count_total!=total_length) printf("afterdifferent\n");
    // printqueue(q_zero_segs);

    // if(segments.size()>=1){
    //   printf("first elem: (%d,%d,%d)\n", segments[0].l, segments[0].r, segments[0].is_zero);
    // }
    sum_values += sumval_in_seg;
    pref_nzeros += nzeros_in_seg;
    return CompCode::COMP_SUCCESS;
  } ;

  u32 total_length = 0;
  u32 nnz = nElems;
  for(u32 i=0;i<nElems; ++i /* in each branch the end of a contiguous segment is reached*/){
    if(src[i]){
      u32 start = i;
      while(i+1<nElems && src[i+1]) ++i;
      // printf("push (%d,%d)\n", start, i);
      if(!segments.size() || segments.back().is_zero ){
        segments.push_back(Seg{start, i, false});
        total_length += i-start+1;
        // printf("push back: %d,%d,%d\n", 0, i, false);
      }else{
        total_length += i-segments.back().r;
        segments.back().r = i;
      }
      // printf("total_length now=%d\n", total_length);
      // for(auto s:segments){
      //   printf("slen=%d\n", s.getLength());
      // }
      while(total_length >= SegLen){
        CompCode gather_result = Consume(total_length);
        if(gather_result != CompCode::COMP_SUCCESS){
          return gather_result;
        }
      }
    }else{
      u32 start = i;
      while(i+1<nElems && !src[i+1]) ++i;
      if(i-start+1>=nz_threshold){
        segments.push_back(Seg{start, i, true});
        nnz -= i-start+1;
        q_zero_segs.push({i-start+1, segments.size()-1});
        // printf("pushque %d\n", segments.size()-1);
        if(q_zero_segs.size() > maxNsegs_perseg){
          auto [len, id] = q_zero_segs.top();
          // printf("segment %d is evicted\n", id);
          segments[id].is_zero = false;
          nnz += len;
          total_length += len;
          // printf("evicted adding total_len+=%d\n", len);
          q_zero_segs.pop();
        }
      }
      else{
        if(!segments.size() || segments.back().is_zero){
          segments.emplace_back(start, i, false);
          total_length += i-start+1;
          // printf("emplace back: %d,%d,%d\n", start, i, false);
        }else{
          total_length += i-segments.back().r;
          segments.back().r = i;
        }
      }
      // "i" is at the end of some segment
    }
  }
  
  while(total_length){
    // printf("processing remaining totallength=%d\n", total_length);
    CompCode gather_result = Consume(total_length);
    // printf("after processing remaining totallength=%d\n", total_length);
    if(gather_result!=CompCode::COMP_SUCCESS){
      return gather_result;
    }
  }
  tgt_anchors.emplace_back(sum_values, pref_nzeros, tgt_data_bytes.size());

  this->nbytes_control = tgt_control.size();
  // printf("controlsize=%d, control[1033], %d, %02x, %p", tgt_control.size(), 1033/8, tgt_control[1033/8], &tgt_control[1033/8]);
  this->control = copy_vec(tgt_control);
  this->bytes_of_data = tgt_data_bytes.size();
  this->data = copy_vec(tgt_data_bytes);
  // this->anchor = copy_vec(tgt_anchors);
  this->anchor_data = (u32*) malloc( tgt_anchors.size() * 3 * sizeof(u32));
  for(int i=0;i<(int)tgt_anchors.size();i++){
    auto [val, nz, pos] = tgt_anchors[i];
    GetValue(anchor_data, i) = val;
    GetPreNz(anchor_data, i) = nz;
    GetSrcPs(anchor_data, i) = pos;
  }

  this->num_segs = tgt_anchors.size()-1;
  this->num_values = nnz;

  // for(int i=0;i<num_values;i++){
  //   printf("%d", (tgt_control[i/8]>>(i&7))&1);
  // }
  
  return COMP_SUCCESS;

}

