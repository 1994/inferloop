#include <metal_stdlib>
using namespace metal;
struct Params {
  uint n,token,position,capacity,a,b,c,d,e,f,g,h;
  float epsilon,offset,theta,extra;
  uint rows,start_row,weight_formats,reserved;
};
#define BUFFERS device const float* x [[buffer(0)]], device const float* w [[buffer(1)]], \
device const float* v [[buffer(2)]], device const float* z [[buffer(3)]], \
device const float* bias [[buffer(4)]], device float* state [[buffer(5)]], \
device float* out [[buffer(6)]], constant Params& p [[buffer(7)]], \
device const uint* page_table [[buffer(8)]], device const uint* tokens [[buffer(9)]]
#define ARGS BUFFERS, uint t [[thread_position_in_grid]], \
uint lane [[thread_index_in_simdgroup]], uint lanes [[threads_per_simdgroup]]
inline float weight(device const float* data, uint at, uint input, constant Params& p) {
  uint format=(p.weight_formats>>(input*2))&3;
  if(format==1) return as_type<float>(uint(reinterpret_cast<device const ushort*>(data)[at])<<16);
  if(format==2) return float(reinterpret_cast<device const half*>(data)[at]);
  return data[at];
}
inline float sig(float x) { return x>=0 ? 1.0f/(1.0f+exp(-x)) : exp(x)/(1.0f+exp(x)); }
inline float sil(float x) { return x*sig(x); }
kernel void embedding(ARGS) {
  if(t<p.n*p.rows) out[t]=weight(x,tokens[p.position+t/p.n]*p.n+t%p.n,0,p);
}
// One SIMD group cooperatively reduces a matrix row. All rows reuse resident weights.
kernel void linear(ARGS) {
  uint cell=t/lanes;
  if(cell>=p.n*p.rows) return;
  uint row=cell/p.n+p.start_row, col=cell%p.n;
  float sum=0;
  for(uint i=lane;i<p.a;i+=lanes) sum+=x[row*p.a+i]*weight(w,col*p.a+i,1,p);
  sum=simd_sum(sum);
  if(lane==0) out[cell]=sum;
}
// 4 token rows x 4 output columns; shared tiles reuse both operands.
// Each SIMD lane preserves the same K-strided accumulation order as decode.
kernel void linear_prefill(BUFFERS, uint2 position [[thread_position_in_grid]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]], uint local [[thread_index_in_threadgroup]]) {
  threadgroup float activations[4*32];
  threadgroup float weights[4*32];
  uint column=group.x*4+local/32, row_start=group.y*4;
  float sums[4]={0,0,0,0};
  for(uint base=0;base<p.a;base+=32) {
    uint input_row=row_start+local/32, k=base+local%32;
    activations[local]=input_row<p.rows && k<p.a ? x[(input_row+p.start_row)*p.a+k] : 0;
    weights[local]=column<p.n && k<p.a ? weight(w,column*p.a+k,1,p) : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint row=0;row<4;row++) sums[row]+=activations[row*32+lane]*weights[local];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for(uint row=0;row<4;row++) {
    float sum=simd_sum(sums[row]);
    if(lane==0 && column<p.n && row_start+row<p.rows) out[(row_start+row)*p.n+column]=sum;
  }
}
kernel void norm(ARGS) {
  uint start=t*p.a; if(start>=p.n*p.rows) return;
  float sum=0; for(uint i=0;i<p.a;i++) sum+=x[start+i]*x[start+i];
  float scale=rsqrt(sum/float(p.a)+p.epsilon);
  for(uint i=0;i<p.a;i++) out[start+i]=x[start+i]*scale*(weight(w,i,1,p)+p.offset);
}
kernel void split(ARGS) {
  if(t>=p.n*p.rows) return;
  uint row=t/p.n, col=t%p.n;
  out[t]=x[row*(p.n/p.c)*p.a+(col/p.c)*p.a+p.b+col%p.c];
}
kernel void rope(ARGS) {
  if(t>=p.n*p.rows) return;
  uint i=t%p.n%p.a, start=t-i, halfdim=p.b/2;
  if(i>=p.b) { out[t]=x[t]; return; }
  uint j=i%halfdim;
  float angle=float(p.position+t/p.n)/pow(p.theta,float(2*j)/float(p.b));
  float s=sin(angle), c=cos(angle);
  out[t]= i<halfdim ? x[start+j]*c-x[start+j+halfdim]*s : x[start+j+halfdim]*c+x[start+j]*s;
}
kernel void kv_append(ARGS) {
  if(t>=p.a*p.rows) return;
  uint pos=p.position+t/p.a, col=t%p.a;
  uint row=page_table[pos/p.e]*p.e+pos%p.e;
  state[row*p.a+col]=w[t]; state[p.f*p.e*p.a+row*p.a+col]=v[t];
}
kernel void attention(ARGS) {
  if(t>=p.a*p.rows) return;
  uint batch=t/p.a, head=t%p.a, kh=head/(p.a/p.b), width=p.b*p.c;
  uint q=batch*p.n+head*p.c, position=p.position+batch;
  uint first=p.d>0 && position+1>p.d ? position+1-p.d : 0;
  float maxscore=-INFINITY, scale=rsqrt(float(p.c));
  for(uint pos=first;pos<=position;pos++) {
    uint row=page_table[pos/p.e]*p.e+pos%p.e;
    float score=0; for(uint d=0;d<p.c;d++) score+=x[q+d]*state[row*width+kh*p.c+d];
    maxscore=max(maxscore,score*scale);
  }
  float denominator=0;
  for(uint d=0;d<p.c;d++) out[q+d]=0;
  for(uint pos=first;pos<=position;pos++) {
    uint row=page_table[pos/p.e]*p.e+pos%p.e;
    float score=0; for(uint d=0;d<p.c;d++) score+=x[q+d]*state[row*width+kh*p.c+d];
    float prob=exp(score*scale-maxscore); denominator+=prob;
    for(uint d=0;d<p.c;d++) out[q+d]+=prob*state[p.f*p.e*width+row*width+kh*p.c+d];
  }
  for(uint d=0;d<p.c;d++) out[q+d]/=denominator;
}
// Every channel owns its history and advances it in token order within this dispatch.
kernel void conv(ARGS) {
  if(t>=p.a) return;
  uint start=t*(p.b-1);
  for(uint row=0;row<p.rows;row++) {
    float sum=x[row*p.a+t]*weight(w,t*p.b+p.b-1,1,p);
    for(uint i=0;i<p.b-1;i++) sum+=state[start+i]*weight(w,t*p.b+i,1,p);
    out[row*p.a+t]=sil(sum);
    for(uint i=0;i+1<p.b-1;i++) state[start+i]=state[start+i+1];
    if(p.b>1) state[start+p.b-2]=x[row*p.a+t];
  }
}
// Every value head owns its recurrent state; token order never crosses a dispatch boundary.
// Value head t reads key head t/e%a: e adjacent value heads share a key head (grouped order),
// or e is 1 and the key heads repeat (tiled order).
kernel void delta(ARGS) {
  if(t>=p.b) return;
  uint kh=t/p.e%p.a, ks=p.a*p.c, base=t*p.c*p.d, channels=2*ks+p.b*p.d;
  for(uint row=0;row<p.rows;row++) {
    uint qstart=row*channels+kh*p.c, kstart=row*channels+ks+kh*p.c;
    uint vstart=row*channels+2*ks+t*p.d;
    float qsum=1e-6f, ksum=1e-6f;
    for(uint i=0;i<p.c;i++) { qsum+=x[qstart+i]*x[qstart+i]; ksum+=x[kstart+i]*x[kstart+i]; }
    float qscale=rsqrt(qsum)*rsqrt(float(p.c)), kscale=rsqrt(ksum);
    float beta=sig(w[row*p.b+t]), a=v[row*p.b+t]+weight(bias,t,4,p);
    float soft=a>20 ? a : log(1.0f+exp(a));
    float decay=exp(-exp(weight(z,t,3,p))*soft);
    for(uint i=0;i<p.c*p.d;i++) state[base+i]*=decay;
    for(uint d=0;d<p.d;d++) {
      float predicted=0;
      for(uint i=0;i<p.c;i++) predicted+=state[base+i*p.d+d]*x[kstart+i]*kscale;
      float correction=(x[vstart+d]-predicted)*beta, result=0;
      for(uint i=0;i<p.c;i++) {
        state[base+i*p.d+d]+=x[kstart+i]*kscale*correction;
        result+=state[base+i*p.d+d]*x[qstart+i]*qscale;
      }
      out[row*p.n+t*p.d+d]=result;
    }
  }
}
kernel void gated_norm(ARGS) {
  uint start=t*p.a; if(start>=p.n*p.rows) return;
  float sum=0; for(uint i=0;i<p.a;i++) sum+=x[start+i]*x[start+i];
  float scale=rsqrt(sum/float(p.a)+p.epsilon);
  for(uint i=0;i<p.a;i++) out[start+i]=x[start+i]*scale*weight(v,i,2,p)*sil(w[start+i]);
}
kernel void unary(ARGS) { if(t<p.n*p.rows) out[t]=p.a==0 ? sil(x[t]) : sig(x[t]); }
kernel void binary(ARGS) { if(t<p.n*p.rows) out[t]=p.a==0 ? x[t]+w[t] : x[t]*w[t]; }
