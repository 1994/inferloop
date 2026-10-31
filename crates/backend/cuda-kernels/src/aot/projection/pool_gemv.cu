// Row-independent BF16 GEMV: one warp per output element with F32 accumulation.
// The reduction order is fixed here and differs from the cuTile tile order, so the numerical
// contract is reference plus tolerance, never bit equality across profiles.
#include <cuda_bf16.h>

extern "C" __global__ void pool_gemv_bf16(
    const float* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ output,
    int columns,
    int outputs) {
  const int row = blockIdx.y;
  const int element = blockIdx.x;
  const float* activations = input + static_cast<long>(row) * columns;
  const __nv_bfloat16* weights = weight + static_cast<long>(element) * columns;
  float sum = 0.0f;
  for (int k = threadIdx.x; k < columns; k += blockDim.x) {
    sum += __bfloat162float(weights[k]) * activations[k];
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (threadIdx.x == 0) output[static_cast<long>(row) * outputs + element] = sum;
}
