// One warp per output row, F32 accumulation, canonical order independent of tile geometry.
extern "C" __global__ void projection_dense_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ out,
    int columns) {
  const int row = blockIdx.x;
  const float* weights = w + static_cast<long>(row) * columns;
  float sum = 0.0f;
  for (int k = threadIdx.x; k < columns; k += blockDim.x) {
    sum += weights[k] * x[k];
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (threadIdx.x == 0) out[row] = sum;
}
