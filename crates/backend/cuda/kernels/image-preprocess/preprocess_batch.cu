// Experimental CUDA image batch preprocessing; buffers and geometry are checked by the host.
#include <stdint.h>
extern "C" __global__ void mm_horizontal_batch(
    const uint8_t* source, const int32_t* index, const int32_t* weight,
    uint8_t* intermediate, int32_t batch, int32_t source_height, int32_t source_width,
    int32_t target_width, int32_t taps, int32_t denominator) {
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t per_image = (uint64_t)source_height * target_width * 3;
  if (i >= per_image * batch) return;
  const uint64_t sample = i / per_image;
  const uint64_t local = i % per_image;
  const int32_t c = local % 3;
  const int32_t x = (local / 3) % target_width;
  const int32_t y = local / ((uint64_t)target_width * 3);
  int64_t sum = 0;
  for (int32_t j = 0; j < taps; ++j) {
    const uint64_t table = (uint64_t)x * taps + j;
    const int32_t source_x = index[table];
    sum += (int64_t)source[((sample * source_height + y) * source_width + source_x) * 3 + c] * weight[table];
  }
  int64_t value = (sum + denominator / 2) / denominator;
  if (value < 0) value = 0;
  if (value > 255) value = 255;
  intermediate[i] = (uint8_t)value;
}
extern "C" __global__ void mm_vertical_patch_batch(
    const uint8_t* intermediate, const int32_t* index, const int32_t* weight,
    float* output, int32_t batch, int32_t source_height, int32_t target_height, int32_t target_width,
    int32_t taps, int32_t denominator, int32_t padded_patches,
    float mean0, float mean1, float mean2, float std0, float std1, float std2) {
  const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint64_t per_image = (uint64_t)padded_patches * 1536;
  if (i >= per_image * batch) return;
  const uint64_t sample = i / per_image;
  const uint64_t local = i % per_image;
  const uint64_t patch = local / 1536;
  const int32_t grid_width = target_width / 16;
  const uint64_t real_patches = (uint64_t)(target_height / 16) * grid_width;
  if (patch >= real_patches) { output[i] = 0.0f; return; }
  const int32_t v = local % 1536;
  const int32_t c = v / 512;
  const int32_t dy = (v / 16) % 16;
  const int32_t dx = v % 16;
  const int32_t mx = patch % 2;
  const int32_t my = (patch / 2) % 2;
  const int32_t gx = (patch / 4) % (grid_width / 2);
  const int32_t gy = patch / (grid_width * 2);
  const int32_t x = (gx * 2 + mx) * 16 + dx;
  const int32_t y = (gy * 2 + my) * 16 + dy;
  int64_t sum = 0;
  for (int32_t j = 0; j < taps; ++j) {
    const uint64_t table = (uint64_t)y * taps + j;
    const int32_t source_y = index[table];
    sum += (int64_t)intermediate[((sample * source_height + source_y) * target_width + x) * 3 + c] * weight[table];
  }
  int64_t value = (sum + denominator / 2) / denominator;
  if (value < 0) value = 0;
  if (value > 255) value = 255;
  const float means[3] = {mean0, mean1, mean2};
  const float stds[3] = {std0, std1, std2};
  output[i] = ((float)value - means[c] * 255.0f) / (stds[c] * 255.0f);
}
