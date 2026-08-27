#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/vec512/vec512_float8.h>
#include <ATen/native/CPUBlas.h>
#include <ATen/native/EmbeddingBag.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Unroll.h>
#include <torch/all.h>
#include "utils.h"

#define QTYPE_DISPATCH(TYPE, ...)                                              \
  [&]() {                                                                      \
    switch (TYPE) {                                                            \
    case c10::ScalarType::Float8_e4m3fn: {                                     \
      using data_t = at::Float8_e4m3fn;                                        \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case c10::ScalarType::Char: {                                              \
      using data_t = int8_t;                                                   \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    default:                                                                   \
      TORCH_CHECK(false, "scaled_embeding_bag: unsupport qtype");              \
    }                                                                          \
  }()

#define OUTTYPE_DISPATCH(TYPE, ...)                                            \
  [&]() {                                                                      \
    switch (TYPE) {                                                            \
    case c10::ScalarType::Float: {                                             \
      using output_t = float;                                                  \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case c10::ScalarType::Char: {                                              \
      using output_t = int8_t;                                                 \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    case c10::ScalarType::Float8_e4m3fn: {                                     \
      using output_t = at::Float8_e4m3fn;                                      \
      return __VA_ARGS__();                                                    \
    }                                                                          \
    default:                                                                   \
      TORCH_CHECK(false, "scaled_embedding_bag: unsupported output type");     \
    }                                                                          \
  }()

namespace torchao {

namespace {

#if defined(CPU_CAPABILITY_AVX512)
using CHUNK =
    std::tuple<__m512, __m512, __m512, __m512, __m512, __m512, __m512, __m512>;
static inline __m512 _mm512_load_e4m3_cvt_ps(const at::Float8_e4m3fn *x) {
  __m512 o;
  __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(x));
  at::vec::CPU_CAPABILITY::cvtfp8e4m3_fp32(v, o);
  return o;
}

static inline __m512 _mm512_cvt_s8_ps(__m128i x) {
  return _mm512_cvt_roundepi32_ps(
      _mm512_cvtepi8_epi32(x), (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

static inline CHUNK load_chunk(const at::Float8_e4m3fn *x) {
  __m512 x0, x1, x2, x3, x4, x5, x6, x7;
  x0 = _mm512_load_e4m3_cvt_ps(x + 0);
  x1 = _mm512_load_e4m3_cvt_ps(x + 16);
  x2 = _mm512_load_e4m3_cvt_ps(x + 32);
  x3 = _mm512_load_e4m3_cvt_ps(x + 48);
  x4 = _mm512_load_e4m3_cvt_ps(x + 64);
  x5 = _mm512_load_e4m3_cvt_ps(x + 80);
  x6 = _mm512_load_e4m3_cvt_ps(x + 96);
  x7 = _mm512_load_e4m3_cvt_ps(x + 112);
  return {x0, x1, x2, x3, x4, x5, x6, x7};
}

static inline CHUNK load_chunk(const int8_t *x) {
  __m512i x00, x64;
  __m512 x0, x1, x2, x3, x4, x5, x6, x7;
  x00 = _mm512_load_si512(x);
  x64 = _mm512_load_si512(x + 64);
  x0 = _mm512_cvt_s8_ps(_mm512_extracti32x4_epi32(x00, 0));
  x1 = _mm512_cvt_s8_ps(_mm512_extracti32x4_epi32(x00, 1));
  x2 = _mm512_cvt_s8_ps(_mm512_extracti32x4_epi32(x00, 2));
  x3 = _mm512_cvt_s8_ps(_mm512_extracti32x4_epi32(x00, 3));
  x4 = _mm512_cvt_s8_ps(_mm512_extracti32x4_epi32(x64, 0));
  x5 = _mm512_cvt_s8_ps(_mm512_extracti32x4_epi32(x64, 1));
  x6 = _mm512_cvt_s8_ps(_mm512_extracti32x4_epi32(x64, 2));
  x7 = _mm512_cvt_s8_ps(_mm512_extracti32x4_epi32(x64, 3));
  return {x0, x1, x2, x3, x4, x5, x6, x7};
}

static inline void store_chunk(float *output, CHUNK chunk) {
  __m512 x0, x1, x2, x3, x4, x5, x6, x7;
  std::tie(x0, x1, x2, x3, x4, x5, x6, x7) = chunk;
  _mm512_store_ps(output, x0);
  _mm512_store_ps(output + 16, x1);
  _mm512_store_ps(output + 32, x2);
  _mm512_store_ps(output + 48, x3);
  _mm512_store_ps(output + 64, x4);
  _mm512_store_ps(output + 80, x5);
  _mm512_store_ps(output + 96, x6);
  _mm512_store_ps(output + 112, x7);
}

static inline void store_chunk(int8_t *output, CHUNK chunk) {
  __m512i x00, x64;
  __m512i y0, y1, y2, y3, y4, y5, y6, y7;
  __m512 f0, f1, f2, f3, f4, f5, f6, f7;
  std::tie(f0, f1, f2, f3, f4, f5, f6, f7) = chunk;
  y0 = _mm512_cvt_roundps_epi32(
      f0, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  y1 = _mm512_cvt_roundps_epi32(
      f1, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  y2 = _mm512_cvt_roundps_epi32(
      f2, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  y3 = _mm512_cvt_roundps_epi32(
      f3, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  y4 = _mm512_cvt_roundps_epi32(
      f4, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  y5 = _mm512_cvt_roundps_epi32(
      f5, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  y6 = _mm512_cvt_roundps_epi32(
      f6, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  y7 = _mm512_cvt_roundps_epi32(
      f7, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  x00 = _mm512_inserti32x4(x00, _mm512_cvtsepi32_epi8(y0), 0);
  x00 = _mm512_inserti32x4(x00, _mm512_cvtsepi32_epi8(y1), 1);
  x00 = _mm512_inserti32x4(x00, _mm512_cvtsepi32_epi8(y2), 2);
  x00 = _mm512_inserti32x4(x00, _mm512_cvtsepi32_epi8(y3), 3);
  x64 = _mm512_inserti32x4(x64, _mm512_cvtsepi32_epi8(y4), 0);
  x64 = _mm512_inserti32x4(x64, _mm512_cvtsepi32_epi8(y5), 1);
  x64 = _mm512_inserti32x4(x64, _mm512_cvtsepi32_epi8(y6), 2);
  x64 = _mm512_inserti32x4(x64, _mm512_cvtsepi32_epi8(y7), 3);
  _mm512_store_si512(output, x00);
  _mm512_store_si512(output + 64, x64);
}

static inline void store_chunk(at::Float8_e4m3fn *output, CHUNK chunk) {
  __m512 x0, x1, x2, x3, x4, x5, x6, x7;
  std::tie(x0, x1, x2, x3, x4, x5, x6, x7) = chunk;
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 0),
                   at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x0));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 16),
                   at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x1));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 32),
                   at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x2));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 48),
                   at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x3));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 64),
                   at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x4));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 80),
                   at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x5));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 96),
                   at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x6));
  _mm_storeu_si128(reinterpret_cast<__m128i *>(output + 112),
                   at::vec::CPU_CAPABILITY::cvtfp32_fp8e4m3(x7));
}

// Prefetch all cache lines of an embedding row (all blocks).
// emb_bytes = emb_dim * sizeof(data_t). Cache line = 64 bytes.
template <typename data_t>
static inline void _prefetch_emb_row(const data_t *base, int64_t emb_dim) {
  const char *ptr = reinterpret_cast<const char *>(base);
  const int64_t emb_bytes = emb_dim * static_cast<int64_t>(sizeof(data_t));
  for (int64_t off = 0; off < emb_bytes; off += 64) {
    _mm_prefetch(ptr + off, _MM_HINT_T0);
  }
}
#endif

static inline void store_elem(float &out, float input) {
  out = input;
}

static inline void store_elem(int8_t &out, float input) {
  float rounded = std::round(input);
  float clamped = std::max(-128.0f, std::min(127.0f, rounded));
  int32_t int32_value = static_cast<int32_t>(clamped);
  out = static_cast<int8_t>(int32_value);
}

static inline void store_elem(at::Float8_e4m3fn &out, float input) {
  out = static_cast<at::Float8_e4m3fn>(input);
}

// Bundle of the arguments shared by every _scaled_embedding_bag_krnl helper.
// Grouping them keeps the helper signatures small and the shared context in one
// place. `result` is the output base pointer; helpers advance a local copy of
// it per batch entry rather than mutating this struct.
template <typename index_t, typename data_t, typename output_t>
struct EmbBagArgs {
  int64_t bs_begin;
  int64_t bs_end;
  int64_t num_emb;
  int64_t emb_dim;
  index_t last_offset;
  const index_t *indices;
  const index_t *offsets;
  const data_t *weight;
  double scale;
  output_t *result;
  int64_t num_batch;
};

// Half-open [start, end) range of indices contributing to one batch entry.
struct RowSpan {
  int64_t start;
  int64_t end;
};

#if defined(CPU_CAPABILITY_AVX512)
// One 128-wide block of one batch entry: which block, and the rows to sum.
struct BlockCtx {
  int64_t block_id;
  RowSpan span;
};
#endif

// Return the row span for batch entry `b`. The last batch entry may use
// `last_offset` (when set) instead of `offsets[b + 1]` for its end.
template <typename index_t, typename data_t, typename output_t>
static inline RowSpan
_batch_row_span(const int64_t b,
                const EmbBagArgs<index_t, data_t, output_t> &args) {
  const bool is_last_batch =
      (b + 1) == args.num_batch && args.last_offset != -1;
  const int64_t end =
      is_last_batch ? args.last_offset : args.offsets[b + 1];
  return {args.offsets[b], end};
}

#if defined(CPU_CAPABILITY_AVX512)
// How many batch entries ahead to prefetch. Each entry has ~3 rows to fetch
// from a 40M-row table; DRAM latency ~100 ns means we must keep enough
// in-flight requests to hide latency.
constexpr int64_t PREFETCH_DIST = 8;

// Software prefetch for batch entry b+PREFETCH_DIST to overlap DRAM latency
// (~100 ns per random access to large table) with AVX512 compute.
template <typename index_t, typename data_t, typename output_t>
static inline void
_prefetch_batch_ahead(const int64_t b,
                      const EmbBagArgs<index_t, data_t, output_t> &args) {
  const int64_t pref_b = b + PREFETCH_DIST;
  if (pref_b >= args.bs_end) {
    return;
  }
  const RowSpan span = _batch_row_span(pref_b, args);
  for (int64_t pj = span.start; pj < span.end; ++pj) {
    _prefetch_emb_row(args.weight + args.indices[pj] * args.emb_dim,
                      args.emb_dim);
  }
}

// Sum one 128-wide block across a batch entry's rows, scale, and store.
template <typename index_t, typename data_t, typename output_t>
static inline void _scaled_embedding_bag_block_avx512(
    const EmbBagArgs<index_t, data_t, output_t> &args, const BlockCtx ctx,
    const __m512 scale_v, output_t *result) {
  constexpr int64_t block_dim = 128;
  const int64_t block_id = ctx.block_id;
  const RowSpan span = ctx.span;
  const int64_t emb_dim = args.emb_dim;
  const index_t *indices = args.indices;
  const data_t *weight = args.weight;
  __m512 x0, x1, x2, x3, x4, x5, x6, x7;
  __m512 y0, y1, y2, y3, y4, y5, y6, y7;
  // load first indices
  int64_t idx = indices[span.start] * emb_dim + block_dim * block_id;
  output_t *block_result = result + block_dim * block_id;
  std::tie(x0, x1, x2, x3, x4, x5, x6, x7) = load_chunk(weight + idx);
  for (int64_t j = span.start + 1; j < span.end; ++j) {
    // add following idx
    idx = indices[j] * emb_dim + block_dim * block_id;
    std::tie(y0, y1, y2, y3, y4, y5, y6, y7) = load_chunk(weight + idx);
    x0 = _mm512_add_ps(x0, y0);
    x1 = _mm512_add_ps(x1, y1);
    x2 = _mm512_add_ps(x2, y2);
    x3 = _mm512_add_ps(x3, y3);
    x4 = _mm512_add_ps(x4, y4);
    x5 = _mm512_add_ps(x5, y5);
    x6 = _mm512_add_ps(x6, y6);
    x7 = _mm512_add_ps(x7, y7);
  }
  x0 = _mm512_mul_ps(x0, scale_v);
  x1 = _mm512_mul_ps(x1, scale_v);
  x2 = _mm512_mul_ps(x2, scale_v);
  x3 = _mm512_mul_ps(x3, scale_v);
  x4 = _mm512_mul_ps(x4, scale_v);
  x5 = _mm512_mul_ps(x5, scale_v);
  x6 = _mm512_mul_ps(x6, scale_v);
  x7 = _mm512_mul_ps(x7, scale_v);
  // store
  store_chunk(block_result, {x0, x1, x2, x3, x4, x5, x6, x7});
}

template <typename index_t, typename data_t, typename output_t>
static inline void
_scaled_embedding_bag_avx512(const EmbBagArgs<index_t, data_t, output_t> &args) {
  constexpr int64_t block_dim = 128;
  const int64_t num_blocks = args.emb_dim / block_dim;
  const __m512 scale_v = _mm512_set1_ps(args.scale);
  output_t *result = args.result;
  for (int64_t b = args.bs_begin; b < args.bs_end; ++b) {
    _prefetch_batch_ahead(b, args);
    const RowSpan span = _batch_row_span(b, args);
    for (int64_t block_id = 0; block_id < num_blocks; block_id++) {
      _scaled_embedding_bag_block_avx512(args, BlockCtx{block_id, span},
                                         scale_v, result);
    }
    result += args.num_emb * args.emb_dim;
  }
}
#endif

// Scalar fallback: sum a batch entry's rows element-wise, scale, and store.
template <typename index_t, typename data_t, typename output_t>
static inline void
_scaled_embedding_bag_scalar(const EmbBagArgs<index_t, data_t, output_t> &args) {
  const int64_t emb_dim = args.emb_dim;
  const index_t *indices = args.indices;
  const data_t *weight = args.weight;
  output_t *result = args.result;
  for (int64_t b = args.bs_begin; b < args.bs_end; ++b) {
    const RowSpan span = _batch_row_span(b, args);
    for (int64_t d = 0; d < emb_dim; d++) {
      int64_t idx = indices[span.start] * emb_dim;
      float value = float(weight[idx + d]);
      for (int64_t j = span.start + 1; j < span.end; ++j) {
        idx = indices[j] * emb_dim;
        value += float(weight[idx + d]);
      }
      value = value * args.scale;
      store_elem(result[d], value);
    }
    result += args.num_emb * emb_dim;
  }
}

template <typename index_t, typename data_t, typename output_t>
inline void _scaled_embedding_bag_krnl(
    const int64_t bs_begin, const int64_t bs_end, const int64_t num_emb,
    const int64_t emb_dim, const index_t last_offset, const index_t *indices,
    const index_t *offsets, const data_t *weight, const double scale,
    output_t *result, const int64_t num_batch) {
  const EmbBagArgs<index_t, data_t, output_t> args{
      bs_begin, bs_end, num_emb, emb_dim,  last_offset, indices,
      offsets,  weight, scale,   result,   num_batch};
#if defined(CPU_CAPABILITY_AVX512)
  if (kHasAVX512 && emb_dim % 128 == 0) {
    _scaled_embedding_bag_avx512(args);
    return;
  }
#endif
  _scaled_embedding_bag_scalar(args);
}

template <typename index_t, typename data_t, typename output_t>
void _scaled_embedding_bag(output_t *o_ptr, data_t *w_ptr, index_t *indices_ptr,
                           index_t *offsets_ptr, int64_t num_batch,
                           int64_t emb_dim, index_t last_offset, double w_scale,
                           double o_scale) {
  constexpr int64_t b_block = 512;
  const int64_t n_b_blocks = (num_batch - 1) / b_block + 1;
  w_scale /= o_scale;
  const int64_t num_emb = 1;
#pragma omp parallel for collapse(2)
  for (int64_t b = 0; b < n_b_blocks; ++b) {
    for (int64_t n = 0; n < num_emb; ++n) {
      const int64_t bs_begin = b * b_block;
      const int64_t bs_end = std::min(num_batch, (b + 1) * b_block);
      output_t *r = &o_ptr[b * b_block * num_emb * emb_dim + n * emb_dim];
      // avoid offsets not include last batch
      _scaled_embedding_bag_krnl(bs_begin, bs_end, num_emb, emb_dim,
                                 last_offset, indices_ptr, offsets_ptr, w_ptr,
                                 w_scale, r, num_batch);
    }
  }
}

template <typename index_t, typename data_t, typename output_t>
void _scaled_embedding_bag_dispatch_dtype(
    const at::Tensor &qweight, const at::Tensor &indices,
    const at::Tensor &offsets, const at::Tensor &output, int64_t batch_size,
    int64_t emb_dim, index_t last_offset, double w_scale, double o_scale) {
  data_t *qweight_ptr = qweight.data_ptr<data_t>();
  index_t *indices_ptr = indices.data_ptr<index_t>();
  index_t *offsets_ptr = offsets.data_ptr<index_t>();
  output_t *output_ptr = output.data_ptr<output_t>();
  _scaled_embedding_bag<index_t, data_t, output_t>(
      output_ptr, qweight_ptr, indices_ptr, offsets_ptr, batch_size, emb_dim,
      last_offset, w_scale, o_scale);
}

at::Tensor _scaled_embedding_bag_impl(
    const at::Tensor &qweight, const at::Tensor &indices,
    const at::Tensor &offsets, const at::Tensor &w_scales, double o_scale,
    const int64_t mode, bool include_last_offset, at::ScalarType output_dtype) {
  // Only support include_last_offset == True and mode ==
  // at::native::EmbeddingBagMode::SUM
  // TODO: Support more case
  TORCH_CHECK(include_last_offset,
              "_scaled_embedding_bag: only suppport include_last_offset");
  TORCH_CHECK(mode == at::native::EmbeddingBagMode::SUM,
              "_scaled_embedding_bag: only suppport sum mode");
  int64_t batch_size =
      include_last_offset ? offsets.size(0) - 1 : offsets.size(0);
  int64_t emb_dim = qweight.size(1);

  auto index_type = indices.scalar_type();
  auto qtype = qweight.scalar_type();
  float w_scale = w_scales.data_ptr<float>()[0];

  TORCH_CHECK(indices.is_contiguous() && offsets.is_contiguous(),
              "_scaled_embedding_bag: only accept contiguous input");
  TORCH_CHECK(
      offsets.scalar_type() == index_type,
      "_scaled_embedding_bag: index and offset must be of the same type");
  TORCH_CHECK(qweight.is_contiguous(),
              "_scaled_embedding_bag: only accept contiguous weight");
  TORCH_CHECK(qweight.dim() == 2,
              "_scaled_embedding_bag: only accept weight with dim == 2");
  TORCH_CHECK(qtype == c10::ScalarType::Float8_e4m3fn ||
              qtype == c10::ScalarType::Char,
              "_scaled_embedding_bag: only support e4m3fn and int8 weight")
  // handle last offsets
  int64_t last_offset = indices.numel();

  at::Tensor output =
      at::empty({batch_size, emb_dim}, qweight.options().dtype(output_dtype));
  OUTTYPE_DISPATCH(output_dtype, [&] {
    QTYPE_DISPATCH(qtype, [&] {
      AT_DISPATCH_INDEX_TYPES(
          indices.scalar_type(), "_scaled_embedding_bag", [&] {
            _scaled_embedding_bag_dispatch_dtype<index_t, data_t, output_t>(
                qweight, indices, offsets, output, batch_size, emb_dim,
                last_offset, w_scale, o_scale);
          });
    });
  });
  return output;
}

} // anonymous namespace

TORCH_LIBRARY_IMPL(torchao, CPU, m) {
  m.impl("torchao::_scaled_embedding_bag", &_scaled_embedding_bag_impl);
}

} // namespace torchao
