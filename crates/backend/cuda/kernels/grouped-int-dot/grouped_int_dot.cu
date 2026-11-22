// Grouped-int projection of two to eight activation rows through int8 dot products.
//
// Weights stay in the 16-row tile-major layout of the cuTile kernels (`CodePacking` and
// `GroupedIntLayout` in infer-cuda-kernels): per tile and 256-value span, nibble codes at byte
// `(group * 32 + lane) * 16 + row` (lane `l` holds values `64 group + l` and `64 group + 32 + l`),
// an optional high plane after the matrix's nibbles, or one byte per code at `value * 16 + row`;
// one f32 factor (and bias) per term group at `(tile_span * GROUPS + group) * 16 + row`.
//
// One CTA covers TILES tiles (16 outputs each) with KSPLIT groups of TILES warps; group `k` takes
// spans `k, k + KSPLIT, ...`. The wide entries take four tiles and two groups, sharing each
// quantized span among four tiles; the narrow ones (`_n`) one tile and eight groups, for
// projections with too few tiles to fill the device in four-tile CTAs. Per span, a group quantizes its 256 activations of every row to
// int8 in shared memory, one scale `amax / 127` per 32 values (ties to even), with the exact f32
// sum of every 16 values for the biases. Thread (g, half, word) of a tile warp then reads the 16
// code words of lanes `16 half ..` of value group `g` at row word `word`, transposes them in
// registers into one u32 of four lanes per row, and chains four `dp4a` per row and nibble half.
// The integer sums are exact; `code - zero` is applied as `zero * sum(q)`, and each partial is
// scaled by the weight factor and the activation scale in f32.
#include <stdint.h>

namespace {

constexpr int SPAN = 256;
constexpr int TILE_ROWS = 16;
constexpr int WARP = 32;
// Every entry runs eight warps.
constexpr int THREADS = 256;
constexpr int QUADS = SPAN / 4;
constexpr int SCALE_GROUPS = SPAN / 32;
constexpr int SUM_GROUPS = SPAN / 16;

enum Packing { NIBBLE = 0, HIGH1 = 1, HIGH2 = 2, BYTE = 3 };

struct Params {
    float* out;
    const float* x;
    const uint8_t* codes;
    const uint32_t* high;
    const float* factors;
    const float* biases;
    int rows;
    int outputs;
    int columns;
    int zero;
};

template <int R, int KSPLIT>
struct Quantized {
    uint32_t q[KSPLIT][R][QUADS];
    float scale[KSPLIT][R][SCALE_GROUPS];
    float sum[KSPLIT][R][SUM_GROUPS];
};

__device__ __forceinline__ float reduce_max8(float value) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
    return fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 4));
}

__device__ __forceinline__ uint32_t pack(float value, float inverse, int shift) {
    return (static_cast<uint32_t>(__float2int_rn(value * inverse)) & 0xffu) << shift;
}

// Quantize span `span` of every row into slot `k`; rows past `rows` read as zero.
template <int R, int KSPLIT, int GROUP_THREADS>
__device__ void quantize(const Params& p, Quantized<R, KSPLIT>& s, int k, int span, int thread) {
    for (int index = thread; index < R * QUADS; index += GROUP_THREADS) {
        const int r = index / QUADS;
        const int quad = index % QUADS;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (r < p.rows) {
            v = *reinterpret_cast<const float4*>(
                p.x + static_cast<size_t>(r) * p.columns + span * SPAN + quad * 4);
        }
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        amax = reduce_max8(amax);
        const float inverse = amax > 0.f ? 127.f / amax : 0.f;
        s.q[k][r][quad] = pack(v.x, inverse, 0) | pack(v.y, inverse, 8) | pack(v.z, inverse, 16) |
                          pack(v.w, inverse, 24);
        float total = (v.x + v.y) + (v.z + v.w);
        total += __shfl_xor_sync(0xffffffffu, total, 1);
        total += __shfl_xor_sync(0xffffffffu, total, 2);
        if (quad % 8 == 0) {
            s.scale[k][r][quad / 8] = amax / 127.f;
        }
        if (quad % 4 == 0) {
            s.sum[k][r][quad / 4] = total;
        }
    }
}

// Rows `4 word + b` of four consecutive lanes, one byte per lane, from four lane words.
__device__ __forceinline__ void transpose(const uint32_t (&lanes)[4], uint32_t (&rows)[4]) {
    const uint32_t a = __byte_perm(lanes[0], lanes[1], 0x5140);
    const uint32_t b = __byte_perm(lanes[0], lanes[1], 0x7362);
    const uint32_t c = __byte_perm(lanes[2], lanes[3], 0x5140);
    const uint32_t d = __byte_perm(lanes[2], lanes[3], 0x7362);
    rows[0] = __byte_perm(a, c, 0x5410);
    rows[1] = __byte_perm(a, c, 0x7632);
    rows[2] = __byte_perm(b, d, 0x5410);
    rows[3] = __byte_perm(b, d, 0x7632);
}

// Byte `g` of four lane words, one per lane.
__device__ __forceinline__ uint32_t gather(const uint32_t (&lanes)[4], uint32_t selector) {
    return __byte_perm(__byte_perm(lanes[0], lanes[1], selector),
                       __byte_perm(lanes[2], lanes[3], selector), 0x5410);
}

// Codes of the thread's lanes for rows `4 word + b`: `low[b][j]` holds lanes `16 half + 4j ..`
// of the low values, `high[b][j]` of the values 32 above them, one code per byte. Nibble and split
// codes are below 64, so they read the same as signed bytes; byte codes are returned as
// `code - 128`, which `dp4a` multiplies as signed bytes.
template <int PACK>
__device__ __forceinline__ void decode(const Params& p, int tile_span, int g, int half, int word,
                                       uint32_t (&low)[4][4], uint32_t (&high)[4][4]) {
    const int first = half * 16;
    if constexpr (PACK == BYTE) {
        const uint32_t* words =
            reinterpret_cast<const uint32_t*>(p.codes) + static_cast<size_t>(tile_span) * SPAN * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t lo[4];
            uint32_t hi[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int value = g * 64 + first + 4 * j + i;
                lo[i] = __ldg(words + value * 4 + word);
                hi[i] = __ldg(words + (value + 32) * 4 + word);
            }
            uint32_t rows[4];
            transpose(lo, rows);
#pragma unroll
            for (int b = 0; b < 4; ++b) low[b][j] = rows[b] ^ 0x80808080u;
            transpose(hi, rows);
#pragma unroll
            for (int b = 0; b < 4; ++b) high[b][j] = rows[b] ^ 0x80808080u;
        }
        return;
    }
    const uint32_t* nibbles = reinterpret_cast<const uint32_t*>(p.codes) +
                              static_cast<size_t>(tile_span) * (SPAN / 2) * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t lanes[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            lanes[i] = __ldg(nibbles + (g * 32 + first + 4 * j + i) * 4 + word);
        }
        uint32_t rows[4];
        transpose(lanes, rows);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            low[b][j] = rows[b] & 0x0f0f0f0fu;
            high[b][j] = (rows[b] >> 4) & 0x0f0f0f0fu;
        }
    }
    if constexpr (PACK == HIGH1 || PACK == HIGH2) {
        constexpr int BITS = PACK == HIGH1 ? 1 : 2;
        constexpr int WIDE = 32 * BITS;
        constexpr uint32_t MASK = BITS == 1 ? 0x01010101u : 0x03030303u;
        const uint32_t* plane = p.high + static_cast<size_t>(tile_span) * WIDE * 4;
        // Value `v` of row `4w + b` holds its high bits at bit `4 BITS k + BITS b`, `k = v / WIDE`,
        // of word `(v % WIDE, w)`; for both widths byte `g` of the word covers this group.
        const uint32_t pick = static_cast<uint32_t>(g | (g + 4) << 4);
#pragma unroll
        for (int part = 0; part < 2; ++part) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t lanes[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int lane = first + 4 * j + i + (BITS == 2 ? 32 * part : 0);
                    lanes[i] = __ldg(plane + lane * 4 + word);
                }
                const uint32_t bytes = gather(lanes, pick);
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    // BITS 2: byte g holds rows b at 2b for the part's values; BITS 1: the low
                    // nibble of byte g holds the low values' row bits, the high nibble the high.
                    const int shift = BITS == 2 ? 2 * b : 4 * part + b;
                    const uint32_t top = ((bytes >> shift) & MASK) << 4;
                    if (part == 0) {
                        low[b][j] |= top;
                    } else {
                        high[b][j] |= top;
                    }
                }
            }
        }
    }
}

template <int PACK, bool BIAS, int R, int TILES, int KSPLIT>
__device__ void project(const Params& p) {
    static_assert(TILES * KSPLIT * WARP == THREADS, "eight warps per CTA");
    constexpr int GROUP_THREADS = TILES * WARP;
    __shared__ Quantized<R, KSPLIT> s;
    __shared__ float partial[KSPLIT][TILES][TILE_ROWS][R];
    constexpr int GROUP = PACK == HIGH2 ? 16 : 32;
    constexpr int GROUPS = SPAN / GROUP;
    const int thread = threadIdx.x;
    const int warp = thread / WARP;
    const int lane = thread % WARP;
    const int k = warp / TILES;
    const int tile = blockIdx.x * TILES + warp % TILES;
    const int tiles = p.outputs / TILE_ROWS;
    const int word = lane % 4;
    const int half = (lane / 4) % 2;
    const int g = lane / 8;
    const int spans = p.columns / SPAN;
    // Byte codes arrive as `code - 128`; the rest of the zero point stays.
    const int zero = PACK == BYTE ? p.zero - 128 : p.zero;
    float acc[4][R];
#pragma unroll
    for (int b = 0; b < 4; ++b)
#pragma unroll
        for (int r = 0; r < R; ++r) acc[b][r] = 0.f;

    for (int base = 0; base < spans; base += KSPLIT) {
        const int span = base + k;
        __syncthreads();
        if (span < spans) {
            quantize<R, KSPLIT, GROUP_THREADS>(p, s, k, span, thread % GROUP_THREADS);
        }
        __syncthreads();
        if (span >= spans || tile >= tiles) continue;
        const int tile_span = tile * spans + span;
        uint32_t low[4][4];
        uint32_t high[4][4];
        decode<PACK>(p, tile_span, g, half, word, low, high);
        const int group_low = (g * 64 + half * 16) / GROUP;
        const int group_high = (g * 64 + 32 + half * 16) / GROUP;
        const float4 f_low = *reinterpret_cast<const float4*>(
            p.factors + (static_cast<size_t>(tile_span) * GROUPS + group_low) * TILE_ROWS + 4 * word);
        const float4 f_high = *reinterpret_cast<const float4*>(
            p.factors + (static_cast<size_t>(tile_span) * GROUPS + group_high) * TILE_ROWS + 4 * word);
        float4 b_low = make_float4(0.f, 0.f, 0.f, 0.f);
        float4 b_high = b_low;
        if constexpr (BIAS) {
            b_low = *reinterpret_cast<const float4*>(
                p.biases + (static_cast<size_t>(tile_span) * GROUPS + group_low) * TILE_ROWS + 4 * word);
            b_high = *reinterpret_cast<const float4*>(
                p.biases + (static_cast<size_t>(tile_span) * GROUPS + group_high) * TILE_ROWS + 4 * word);
        }
        const float fl[4] = {f_low.x, f_low.y, f_low.z, f_low.w};
        const float fh[4] = {f_high.x, f_high.y, f_high.z, f_high.w};
        const float bl[4] = {b_low.x, b_low.y, b_low.z, b_low.w};
        const float bh[4] = {b_high.x, b_high.y, b_high.z, b_high.w};
        const int quad_low = (g * 64 + half * 16) / 4;
        const int quad_high = quad_low + 8;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            uint32_t ql[4];
            uint32_t qh[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                ql[j] = s.q[k][r][quad_low + j];
                qh[j] = s.q[k][r][quad_high + j];
            }
            int sum_low = 0;
            int sum_high = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sum_low = __dp4a(0x01010101, static_cast<int>(ql[j]), sum_low);
                sum_high = __dp4a(0x01010101, static_cast<int>(qh[j]), sum_high);
            }
            const float s_low = s.scale[k][r][quad_low / 8];
            const float s_high = s.scale[k][r][quad_high / 8];
            const float x_low = s.sum[k][r][quad_low / 4];
            const float x_high = s.sum[k][r][quad_high / 4];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                int d_low = -zero * sum_low;
                int d_high = -zero * sum_high;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    d_low = __dp4a(static_cast<int>(low[b][j]), static_cast<int>(ql[j]), d_low);
                    d_high = __dp4a(static_cast<int>(high[b][j]), static_cast<int>(qh[j]), d_high);
                }
                float term = fl[b] * (s_low * __int2float_rn(d_low)) +
                             fh[b] * (s_high * __int2float_rn(d_high));
                if constexpr (BIAS) {
                    term -= bl[b] * x_low + bh[b] * x_high;
                }
                acc[b][r] += term;
            }
        }
    }
    // Threads sharing a row word differ in `g` and `half`: lane bits 2..4.
#pragma unroll
    for (int b = 0; b < 4; ++b)
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float v = acc[b][r];
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            v += __shfl_xor_sync(0xffffffffu, v, 8);
            v += __shfl_xor_sync(0xffffffffu, v, 16);
            acc[b][r] = v;
        }
    if (lane < 4) {
#pragma unroll
        for (int b = 0; b < 4; ++b)
#pragma unroll
            for (int r = 0; r < R; ++r) partial[k][warp % TILES][4 * lane + b][r] = acc[b][r];
    }
    __syncthreads();
    for (int index = thread; index < TILES * TILE_ROWS * R; index += THREADS) {
        const int r = index / (TILES * TILE_ROWS);
        const int local = index % (TILES * TILE_ROWS);
        const int t = local / TILE_ROWS;
        const int row = local % TILE_ROWS;
        const int out_tile = blockIdx.x * TILES + t;
        if (r >= p.rows || out_tile >= tiles) continue;
        float total = 0.f;
#pragma unroll
        for (int kk = 0; kk < KSPLIT; ++kk) total += partial[kk][t][row][r];
        p.out[static_cast<size_t>(r) * p.outputs + out_tile * TILE_ROWS + row] = total;
    }
}

}  // namespace

// Parameters of every entry, in launch order.
#define GROUPED_INT_DOT_PARAMS                                                                 \
    float *out, const float *x, const uint8_t *codes, const uint32_t *high,                    \
        const float *factors, const float *biases, int rows, int outputs, int columns, int zero

#define GROUPED_INT_DOT_BODY(PACK, BIAS, R, TILES, KSPLIT)                                       \
    project<PACK, BIAS, R, TILES, KSPLIT>(                                                      \
        Params{out, x, codes, high, factors, biases, rows, outputs, columns, zero})

extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_nibble_zero_r2(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(NIBBLE, false, 2, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_nibble_zero_r2_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(NIBBLE, false, 2, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_nibble_zero_r4(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(NIBBLE, false, 4, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_nibble_zero_r4_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(NIBBLE, false, 4, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_nibble_zero_r8(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(NIBBLE, false, 8, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_nibble_zero_r8_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(NIBBLE, false, 8, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_nibble_bias_r2(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(NIBBLE, true, 2, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_nibble_bias_r2_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(NIBBLE, true, 2, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_nibble_bias_r4(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(NIBBLE, true, 4, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_nibble_bias_r4_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(NIBBLE, true, 4, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_nibble_bias_r8(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(NIBBLE, true, 8, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_nibble_bias_r8_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(NIBBLE, true, 8, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high1_zero_r2(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH1, false, 2, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high1_zero_r2_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH1, false, 2, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high1_zero_r4(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH1, false, 4, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high1_zero_r4_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH1, false, 4, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high1_zero_r8(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH1, false, 8, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high1_zero_r8_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH1, false, 8, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high1_bias_r2(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH1, true, 2, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high1_bias_r2_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH1, true, 2, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high1_bias_r4(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH1, true, 4, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high1_bias_r4_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH1, true, 4, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high1_bias_r8(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH1, true, 8, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high1_bias_r8_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH1, true, 8, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high2_zero_r2(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH2, false, 2, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high2_zero_r2_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH2, false, 2, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high2_zero_r4(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH2, false, 4, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high2_zero_r4_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH2, false, 4, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high2_zero_r8(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH2, false, 8, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_high2_zero_r8_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(HIGH2, false, 8, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_byte_zero_r2(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(BYTE, false, 2, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_byte_zero_r2_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(BYTE, false, 2, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_byte_zero_r4(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(BYTE, false, 4, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_byte_zero_r4_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(BYTE, false, 4, 1, 8);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_byte_zero_r8(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(BYTE, false, 8, 4, 2);
}
extern "C" __global__ void __launch_bounds__(THREADS)
    grouped_int_dot_byte_zero_r8_n(GROUPED_INT_DOT_PARAMS) {
    GROUPED_INT_DOT_BODY(BYTE, false, 8, 1, 8);
}
