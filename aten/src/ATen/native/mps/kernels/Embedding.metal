#include <ATen/native/mps/kernels/Embedding.h>
#include <c10/metal/atomic.h>
#include <c10/metal/utils.h>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

template <typename O>
static inline void resolve_outer_offsets(
    uint tid,
    constant EmbeddingDenseBackwardParams<uint32_t>& params,
    thread O& grad_offset,
    thread O& indices_offset) {
  grad_offset = 0;
  indices_offset = 0;
  uint remaining = tid;
  uint nd = params.outer_ndim;
  for (uint d = 0; d < nd; ++d) {
    uint dim_idx = nd - 1 - d;
    uint dim_size = params.outer_sizes[dim_idx];
    uint coord = remaining % dim_size;
    remaining /= dim_size;
    grad_offset += static_cast<O>(coord) *
        static_cast<O>(params.grad_outer_strides[dim_idx]);
    indices_offset +=
        static_cast<O>(coord) * static_cast<O>(params.indices_strides[dim_idx]);
  }
}

template <typename I, typename O>
kernel void embedding_dense_backward_count(
    device const I* indices [[buffer(0)]],
    device atomic_uint* counts [[buffer(1)]],
    constant EmbeddingDenseBackwardParams<uint32_t>& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  O grad_offset_unused = 0;
  O indices_offset = 0;
  resolve_outer_offsets<O>(tid, params, grad_offset_unused, indices_offset);
  long weight_idx = static_cast<long>(indices[indices_offset]);
  if (weight_idx == params.padding_idx) {
    return;
  }
  atomic_fetch_add_explicit(counts + weight_idx, 1u, memory_order_relaxed);
}

template <typename T, typename I, typename O>
kernel void embedding_dense_backward(
    device const T* grad [[buffer(0)]],
    device const I* indices [[buffer(1)]],
    constant uint* counts [[buffer(2)]],
    device AtomicType_t<float>* grad_weight [[buffer(3)]],
    constant EmbeddingDenseBackwardParams<uint32_t>& params [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
  uint feature_size = params.feature_size;
  uint index_pos = tid / feature_size;
  uint feature_idx = tid % feature_size;

  O grad_outer_offset = 0;
  O indices_offset = 0;
  resolve_outer_offsets<O>(
      index_pos, params, grad_outer_offset, indices_offset);

  long weight_idx = static_cast<long>(indices[indices_offset]);
  if (weight_idx == params.padding_idx) {
    return;
  }

  O grad_offset = grad_outer_offset +
      static_cast<O>(feature_idx) * static_cast<O>(params.grad_feature_stride);
  float grad_val = static_cast<float>(grad[grad_offset]);
  if (params.scale_grad_by_freq) {
    uint c = counts[weight_idx];
    if (c == 0) {
      return;
    }
    grad_val /= static_cast<float>(c);
  }

  AtomicType<float>::atomic_add(
      grad_weight,
      static_cast<long>(weight_idx) * feature_size + feature_idx,
      grad_val);
}

// Deterministic path over stably sorted indices: each run of equal indices is
// contiguous and keeps the original row order. A thread sums one feature over a
// tile of kEmbeddingBackwardRowsPerTile rows. Runs that fit in the tile are
// written directly. A run that crosses a tile boundary leaves its piece in
// partials[2 * tile] if it began in an earlier tile, or partials[2 * tile + 1]
// if it begins here, and embedding_dense_backward_sorted_carry sums the pieces
// in tile order.
static inline uint first_of(
    device const int* sorted,
    uint lo,
    uint hi,
    int value) {
  while (lo < hi) {
    const uint mid = lo + (hi - lo) / 2;
    if (sorted[mid] < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

static inline uint past_last_of(
    device const int* sorted,
    uint lo,
    uint hi,
    int value) {
  while (lo < hi) {
    const uint mid = lo + (hi - lo) / 2;
    if (sorted[mid] <= value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <typename T>
kernel void embedding_dense_backward_sorted(
    device const T* grad [[buffer(0)]],
    device const int* sorted_indices [[buffer(1)]],
    device const long* positions [[buffer(2)]],
    device T* grad_weight [[buffer(3)]],
    device float* partials [[buffer(4)]],
    constant EmbeddingDenseBackwardSortedParams& params [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]) {
  const uint feature = tid.x;
  const uint tile = tid.y;
  const uint num_indices = params.num_indices;
  const uint feature_size = params.feature_size;
  const uint begin = tile * kEmbeddingBackwardRowsPerTile;
  const uint end = min(begin + kEmbeddingBackwardRowsPerTile, num_indices);
  const bool head_continues =
      begin > 0 && sorted_indices[begin - 1] == sorted_indices[begin];
  const bool tail_continues =
      end < num_indices && sorted_indices[end] == sorted_indices[end - 1];
  for (uint run_begin = begin; run_begin < end;) {
    const int index = sorted_indices[run_begin];
    uint run_end = run_begin + 1;
    while (run_end < end && sorted_indices[run_end] == index) {
      ++run_end;
    }
    const bool starts_here = run_begin > begin || !head_continues;
    const bool ends_here = run_end < end || !tail_continues;
    if (index != params.padding_idx) {
      float scale = 1;
      if (params.scale_grad_by_freq) {
        const uint first =
            starts_here ? run_begin : first_of(sorted_indices, 0, begin, index);
        const uint last = ends_here
            ? run_end
            : past_last_of(sorted_indices, end, num_indices, index);
        scale = 1.0f / static_cast<float>(last - first);
      }
      float acc = 0;
      for (uint i = run_begin; i < run_end; ++i) {
        const uint row = static_cast<uint>(positions[i]);
        acc += static_cast<float>(grad[row * feature_size + feature]) * scale;
      }
      if (starts_here && ends_here) {
        grad_weight[static_cast<uint>(index) * feature_size + feature] =
            static_cast<T>(acc);
      } else {
        partials[(2 * tile + (starts_here ? 1 : 0)) * feature_size + feature] =
            acc;
      }
    }
    run_begin = run_end;
  }
}

template <typename T>
kernel void embedding_dense_backward_sorted_carry(
    device const int* sorted_indices [[buffer(0)]],
    device T* grad_weight [[buffer(1)]],
    device const float* partials [[buffer(2)]],
    constant EmbeddingDenseBackwardSortedParams& params [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
  const uint feature = tid.x;
  const uint tile = tid.y;
  const uint num_indices = params.num_indices;
  const uint feature_size = params.feature_size;
  const uint begin = tile * kEmbeddingBackwardRowsPerTile;
  const uint end = begin + kEmbeddingBackwardRowsPerTile;
  if (end >= num_indices) {
    return;
  }
  // Only the tile where a run crossing its end begins finishes that run
  const int index = sorted_indices[end - 1];
  if (sorted_indices[end] != index || index == params.padding_idx ||
      (begin > 0 && sorted_indices[begin - 1] == index)) {
    return;
  }
  float acc = partials[(2 * tile + 1) * feature_size + feature];
  for (uint next = tile + 1;; ++next) {
    acc += partials[2 * next * feature_size + feature];
    const uint next_end = (next + 1) * kEmbeddingBackwardRowsPerTile;
    if (next_end >= num_indices || sorted_indices[next_end] != index) {
      break;
    }
  }
  grad_weight[static_cast<uint>(index) * feature_size + feature] =
      static_cast<T>(acc);
}

#define REGISTER_EMBEDDING_DENSE_BACKWARD_SORTED(T)                      \
  template [[host_name("embedding_dense_backward_sorted_" #T)]]          \
  kernel void embedding_dense_backward_sorted<T>(                        \
      device const T* grad [[buffer(0)]],                                \
      device const int* sorted_indices [[buffer(1)]],                    \
      device const long* positions [[buffer(2)]],                        \
      device T* grad_weight [[buffer(3)]],                               \
      device float* partials [[buffer(4)]],                              \
      constant EmbeddingDenseBackwardSortedParams& params [[buffer(5)]], \
      uint2 tid [[thread_position_in_grid]]);                            \
  template [[host_name("embedding_dense_backward_sorted_carry_" #T)]]    \
  kernel void embedding_dense_backward_sorted_carry<T>(                  \
      device const int* sorted_indices [[buffer(0)]],                    \
      device T* grad_weight [[buffer(1)]],                               \
      device const float* partials [[buffer(2)]],                        \
      constant EmbeddingDenseBackwardSortedParams& params [[buffer(3)]], \
      uint2 tid [[thread_position_in_grid]])

#define REGISTER_EMBEDDING_DENSE_BACKWARD_COUNT(I, O, OSUFFIX)               \
  template [[host_name("embedding_dense_backward_count_" #I "_" #OSUFFIX)]]  \
  kernel void embedding_dense_backward_count<I, O>(                          \
      device const I* indices [[buffer(0)]],                                 \
      device atomic_uint* counts [[buffer(1)]],                              \
      constant EmbeddingDenseBackwardParams<uint32_t>& params [[buffer(2)]], \
      uint tid [[thread_position_in_grid]])

#define REGISTER_EMBEDDING_DENSE_BACKWARD(T, I, O, OSUFFIX)                  \
  template [[host_name("embedding_dense_backward_" #T "_" #I "_" #OSUFFIX)]] \
  kernel void embedding_dense_backward<T, I, O>(                             \
      device const T* grad [[buffer(0)]],                                    \
      device const I* indices [[buffer(1)]],                                 \
      constant uint* counts [[buffer(2)]],                                   \
      device AtomicType_t<float>* grad_weight [[buffer(3)]],                 \
      constant EmbeddingDenseBackwardParams<uint32_t>& params [[buffer(4)]], \
      uint tid [[thread_position_in_grid]])

REGISTER_EMBEDDING_DENSE_BACKWARD_COUNT(int, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD_COUNT(int, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD_COUNT(long, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD_COUNT(long, ulong, 64);

REGISTER_EMBEDDING_DENSE_BACKWARD(float, int, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(float, int, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD(float, long, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(float, long, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD(half, int, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(half, int, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD(half, long, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(half, long, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD(bfloat, int, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(bfloat, int, ulong, 64);
REGISTER_EMBEDDING_DENSE_BACKWARD(bfloat, long, uint, 32);
REGISTER_EMBEDDING_DENSE_BACKWARD(bfloat, long, ulong, 64);

REGISTER_EMBEDDING_DENSE_BACKWARD_SORTED(float);
REGISTER_EMBEDDING_DENSE_BACKWARD_SORTED(half);
REGISTER_EMBEDDING_DENSE_BACKWARD_SORTED(bfloat);
