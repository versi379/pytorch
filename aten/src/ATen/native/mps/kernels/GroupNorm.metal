#include <ATen/native/mps/kernels/GroupNorm.h>
#include <c10/metal/reduction_utils.h>
#include <c10/metal/utils.h>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

template <typename T>
inline float load_affine_scale(device const T* ptr, uint idx) {
  return float(ptr[idx]);
}

template <>
inline float load_affine_scale(device const void*, uint) {
  return 1;
}

template <typename T>
inline float load_affine_bias(device const T* ptr, uint idx) {
  return float(ptr[idx]);
}

template <>
inline float load_affine_bias(device const void*, uint) {
  return 0;
}

// Merges the count, mean and sum of squared deviations (m2) of a set of
// elements into the running statistics of another (Chan et al.). The rounding
// errors of the running mean and m2 are kept in mean_err and m2_err, so that
// small updates are not lost against large running values.
inline void welford_merge(
    thread float& n,
    thread float& mean,
    thread float& mean_err,
    thread float& m2,
    thread float& m2_err,
    float n_b,
    float mean_b,
    float m2_b) {
  float n_ab = n + n_b;
  // The few-ulp error of fast::divide only scales this one update.
  float w = ::metal::fast::divide(n_b, n_ab);
  float delta = (mean_b - mean) - mean_err;
  float mean_inc = delta * w;
  float new_mean = mean + mean_inc;
  mean_err -= (new_mean - mean) - mean_inc;
  mean = new_mean;
  float m2_inc = fma(delta * delta * n, w, m2_b);
  float new_m2 = m2 + m2_inc;
  m2_err -= (new_m2 - m2) - m2_inc;
  m2 = new_m2;
  n = n_ab;
}

// Merges the (n, mean, m2) of every thread in the threadgroup: each simdgroup
// first, then the simdgroups. Returns (n, mean, m2) of the threadgroup.
inline float3 threadgroup_welford(
    threadgroup float3* data,
    float n,
    float mean,
    float m2,
    unsigned idx,
    unsigned size) {
  float n_s = ::metal::simd_sum(n);
  float mean_s = ::metal::simd_sum(n * mean) / n_s;
  float d = mean - mean_s;
  float m2_s = ::metal::simd_sum(fma(n * d, d, m2));
  if (idx % simdgroup_size == 0) {
    data[idx / simdgroup_size] = float3(n_s, mean_s, m2_s);
  }
  if (size > simdgroup_size) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (idx < simdgroup_size) {
      auto p = idx < (size + simdgroup_size - 1) / simdgroup_size ? data[idx]
                                                                  : float3(0);
      float n_g = ::metal::simd_sum(p.x);
      float mean_g = ::metal::simd_sum(p.x * p.y) / n_g;
      float d_g = p.y - mean_g;
      float m2_g = ::metal::simd_sum(fma(p.x * d_g, d_g, p.z));
      if (idx == 0) {
        data[0] = float3(n_g, mean_g, m2_g);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return data[0];
}

template <
    typename T,
    typename stat_T,
    typename gamma_T,
    typename beta_T,
    typename idx_T>
kernel void group_norm(
    device T* Y [[buffer(0)]],
    device stat_T* mean [[buffer(1)]],
    device stat_T* rstd [[buffer(2)]],
    device const T* X [[buffer(3)]],
    device const gamma_T* gamma [[buffer(4)]],
    device const beta_T* beta [[buffer(5)]],
    constant GroupNormParams<idx_T>& params [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  idx_T group_offset = tgid * params.elements_per_group;
  device const T* x = X + group_offset;
  device T* y = Y + group_offset;

  // Divide the elements of the group between each thread in the threadgroup.
  // First, each thread computes the count, mean and m2 of the elements
  // assigned to it, one block at a time. Values are taken as offsets from the
  // first element of the group, so a mean much larger than the spread keeps
  // its precision. Unlike `sum(x**2) - sum(x)**2 / m`, the merges do not
  // cancel, so an outlier in that first element does not cost the variance
  // its precision either.
  float shift = float(x[0]);
  float n = 0, mean_v = 0, mean_err = 0, m2 = 0, m2_err = 0;

  for (idx_T r = 0; r < params.elements_per_group; r += tptg * BLOCK_SIZE) {
    auto base = r + tid * BLOCK_SIZE;
    if (base + BLOCK_SIZE <= params.elements_per_group) {
      float v[BLOCK_SIZE];
      float block_sum = 0;
#pragma unroll
      for (idx_T i = 0; i < BLOCK_SIZE; i++) {
        v[i] = static_cast<float>(x[base + i]) - shift;
        block_sum += v[i];
      }
      float block_mean = block_sum / BLOCK_SIZE;
      float block_m2 = 0;
#pragma unroll
      for (idx_T i = 0; i < BLOCK_SIZE; i++) {
        float d = v[i] - block_mean;
        block_m2 = fma(d, d, block_m2);
      }
      welford_merge(
          n, mean_v, mean_err, m2, m2_err, BLOCK_SIZE, block_mean, block_m2);
    } else {
      for (auto i = base; i < params.elements_per_group; i++) {
        welford_merge(
            n,
            mean_v,
            mean_err,
            m2,
            m2_err,
            1,
            static_cast<float>(x[i]) - shift,
            0);
      }
    }
  }

  // Second, merge the statistics of all the threads in the threadgroup and
  // calculate the mean and rstd.
  threadgroup float3 local_stats[simdgroup_size];
  auto stats = threadgroup_welford(
      local_stats, n, mean_v + mean_err, m2 + m2_err, tid, tptg);

  float mean_val = shift + stats.y;
  float var_val = stats.z / float(params.elements_per_group);
  float rstd_val = metal::precise::rsqrt(var_val + params.eps);

  // Third, each thread reads its assigned input elements again, applies the
  // normalization and affine transform, and writes the results to the output.
  idx_T channel_base = (tgid % params.num_groups) * params.channels_per_group;

  for (idx_T r = 0; r < params.elements_per_group; r += tptg * BLOCK_SIZE) {
    idx_T base = r + tid * BLOCK_SIZE;
#pragma unroll
    for (idx_T i = 0; i < BLOCK_SIZE; i++) {
      idx_T elem = base + i;
      if (elem < params.elements_per_group) {
        idx_T channel = channel_base + elem / params.HxW;
        float norm = (float(x[elem]) - mean_val) * rstd_val;
        norm = norm * load_affine_scale(gamma, channel) +
            load_affine_bias(beta, channel);
        y[elem] = T(norm);
      }
    }
  }

  // Finally, one thread in the group writes the mean and rstd outputs.
  if (tid == 0) {
    mean[tgid] = stat_T(mean_val);
    rstd[tgid] = stat_T(rstd_val);
  }
}

#define REGISTER_GROUP_NORM(T, stat_T, gamma_T, beta_T, idx_T)               \
  template [[host_name("group_norm_" #T "_" #stat_T "_" #gamma_T "_" #beta_T \
                       "_" #idx_T)]]                                         \
  kernel void group_norm<T, stat_T, gamma_T, beta_T, idx_T>(                 \
      device T * Y [[buffer(0)]],                                            \
      device stat_T * mean [[buffer(1)]],                                    \
      device stat_T * rstd [[buffer(2)]],                                    \
      device const T* X [[buffer(3)]],                                       \
      device const gamma_T* gamma [[buffer(4)]],                             \
      device const beta_T* beta [[buffer(5)]],                               \
      constant GroupNormParams<idx_T>& params [[buffer(6)]],                 \
      uint tg_id [[threadgroup_position_in_grid]],                           \
      uint tid [[thread_position_in_threadgroup]],                           \
      uint tptg [[threads_per_threadgroup]]);

#define REGISTER_GROUP_NORM_IDX_TYPES(T, T_stat, gamma_T, beta_T) \
  REGISTER_GROUP_NORM(T, T_stat, gamma_T, beta_T, uint32_t);      \
  REGISTER_GROUP_NORM(T, T_stat, gamma_T, beta_T, uint64_t);

#define REGISTER_GROUP_NORM_AFFINE_TYPES_INNER(T, T_stat, affine_T) \
  REGISTER_GROUP_NORM_IDX_TYPES(T, T_stat, affine_T, affine_T);     \
  REGISTER_GROUP_NORM_IDX_TYPES(T, T_stat, affine_T, void);         \
  REGISTER_GROUP_NORM_IDX_TYPES(T, T_stat, void, affine_T);

#define REGISTER_GROUP_NORM_AFFINE_TYPES(T, T_stat)          \
  REGISTER_GROUP_NORM_AFFINE_TYPES_INNER(T, T_stat, float);  \
  REGISTER_GROUP_NORM_AFFINE_TYPES_INNER(T, T_stat, half);   \
  REGISTER_GROUP_NORM_AFFINE_TYPES_INNER(T, T_stat, bfloat); \
  REGISTER_GROUP_NORM_IDX_TYPES(T, T_stat, void, void);

REGISTER_GROUP_NORM_AFFINE_TYPES(float, float);
REGISTER_GROUP_NORM_AFFINE_TYPES(half, float);
REGISTER_GROUP_NORM_AFFINE_TYPES(bfloat, float);

REGISTER_GROUP_NORM_AFFINE_TYPES(half, half);
REGISTER_GROUP_NORM_AFFINE_TYPES(bfloat, bfloat);

template <typename T, typename stat_T, typename gamma_T, typename idx_T>
kernel void group_norm_backward_x(
    device T* dX [[buffer(0)]],
    device const T* dY [[buffer(1)]],
    device const T* X [[buffer(2)]],
    device const stat_T* mean [[buffer(3)]],
    device const stat_T* rstd [[buffer(4)]],
    device const gamma_T* gamma [[buffer(5)]],
    constant GroupNormParams<idx_T>& params [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  idx_T group_offset = tgid * params.elements_per_group;
  device const T* x = X + group_offset;
  device const T* dy = dY + group_offset;
  device T* dx = dX + group_offset;

  auto mean_val = float(mean[tgid]);
  auto rstd_val = float(rstd[tgid]);
  idx_T channel_base = (tgid % params.num_groups) * params.channels_per_group;

  // Accumulate `ds = sum(dY * gamma * X)` and `db = sum(dY * gamma)` over all
  // elements in the group.
  float partial_ds = 0;
  float partial_db = 0;
  for (idx_T r = 0; r < params.elements_per_group; r += tptg * BLOCK_SIZE) {
    idx_T base = r + tid * BLOCK_SIZE;
#pragma unroll
    for (idx_T i = 0; i < BLOCK_SIZE; i++) {
      auto elem = base + i;
      if (elem < params.elements_per_group) {
        auto gamma_val =
            load_affine_scale(gamma, channel_base + elem / params.HxW);
        auto dy_val = float(dy[elem]);
        auto x_val = float(x[elem]);
        auto dy_gamma = dy_val * gamma_val;
        partial_ds += dy_gamma * x_val;
        partial_db += dy_gamma;
      }
    }
  }

  // Reduce ds and db across the threadgroup.
  threadgroup float local_ds[simdgroup_size];
  threadgroup float local_db[simdgroup_size];
  auto reduction =
      threadgroup_sum2(local_ds, local_db, partial_ds, partial_db, tid, tptg);
  auto ds_val = reduction[0];
  auto db_val = reduction[1];

  // Compute per-group coefficients
  auto m = float(params.elements_per_group);
  auto c2 = (db_val * mean_val - ds_val) * rstd_val * rstd_val * rstd_val / m;
  auto c3 = -c2 * mean_val - db_val * rstd_val / m;

  // Write dX.
  for (idx_T r = 0; r < params.elements_per_group; r += tptg * BLOCK_SIZE) {
    idx_T base = r + tid * BLOCK_SIZE;
#pragma unroll
    for (idx_T i = 0; i < BLOCK_SIZE; i++) {
      idx_T elem = base + i;
      if (elem < params.elements_per_group) {
        auto c1 = rstd_val *
            load_affine_scale(gamma, channel_base + elem / params.HxW);
        dx[elem] = T(c1 * float(dy[elem]) + c2 * float(x[elem]) + c3);
      }
    }
  }
}

#define REGISTER_GROUP_NORM_BACKWARD_X(T, stat_T, gamma_T, idx_T)           \
  template [[host_name("group_norm_backward_x_" #T "_" #stat_T "_" #gamma_T \
                       "_" #idx_T)]]                                        \
  kernel void group_norm_backward_x<T, stat_T, gamma_T, idx_T>(             \
      device T * dX [[buffer(0)]],                                          \
      device const T* dY [[buffer(1)]],                                     \
      device const T* X [[buffer(2)]],                                      \
      device const stat_T* mean [[buffer(3)]],                              \
      device const stat_T* rstd [[buffer(4)]],                              \
      device const gamma_T* gamma [[buffer(5)]],                            \
      constant GroupNormParams<idx_T>& params [[buffer(6)]],                \
      uint tgid [[threadgroup_position_in_grid]],                           \
      uint tid [[thread_position_in_threadgroup]],                          \
      uint tptg [[threads_per_threadgroup]]);

#define REGISTER_GROUP_NORM_BACKWARD_INDEX_TYPES(T, stat_T, gamma_T) \
  REGISTER_GROUP_NORM_BACKWARD_X(T, stat_T, gamma_T, uint32_t);      \
  REGISTER_GROUP_NORM_BACKWARD_X(T, stat_T, gamma_T, uint64_t);

#define REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(T, stat_T)    \
  REGISTER_GROUP_NORM_BACKWARD_INDEX_TYPES(T, stat_T, float);  \
  REGISTER_GROUP_NORM_BACKWARD_INDEX_TYPES(T, stat_T, half);   \
  REGISTER_GROUP_NORM_BACKWARD_INDEX_TYPES(T, stat_T, bfloat); \
  REGISTER_GROUP_NORM_BACKWARD_INDEX_TYPES(T, stat_T, void);

REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(float, float);
REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(half, float);
REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(bfloat, float);
REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(half, half);
REGISTER_GROUP_NORM_BACKWARD_GAMMA_TYPES(bfloat, bfloat);

template <typename T, typename stat_T, typename affine_T, typename idx_T>
kernel void group_norm_backward_affine(
    device affine_T* dgamma [[buffer(0)]],
    device affine_T* dbeta [[buffer(1)]],
    device const T* dY [[buffer(2)]],
    device const T* X [[buffer(3)]],
    device const stat_T* mean [[buffer(4)]],
    device const stat_T* rstd [[buffer(5)]],
    constant GroupNormParams<idx_T>& params [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  // One threadgroup per channel
  idx_T channel = tgid;
  idx_T group = channel / params.channels_per_group;

  // Accumulate `dg = sum(dy * (x - mean) * rstd)` and `db = sum(dy)` over all
  // dimensions except the channel dimension.
  float partial_dg = 0;
  float partial_db = 0;
  for (idx_T r = 0; r < params.N_times_HxW;
       r += idx_T(tptg) * idx_T(BLOCK_SIZE)) {
    idx_T base = r + idx_T(tid) * idx_T(BLOCK_SIZE);
#pragma unroll
    for (idx_T i = 0; i < idx_T(BLOCK_SIZE); i++) {
      idx_T elem = base + i;
      if (elem < params.N_times_HxW) {
        idx_T batch = elem / params.HxW;
        idx_T ng = batch * params.num_groups + group;
        idx_T idx = batch * params.C * params.HxW + channel * params.HxW +
            elem % params.HxW;
        auto dy_val = float(dY[idx]);
        partial_dg +=
            dy_val * (float(X[idx]) - float(mean[ng])) * float(rstd[ng]);
        partial_db += dy_val;
      }
    }
  }

  threadgroup float local_dg[simdgroup_size];
  threadgroup float local_db[simdgroup_size];
  auto reduction =
      threadgroup_sum2(local_dg, local_db, partial_dg, partial_db, tid, tptg);

  if (tid == 0) {
    dgamma[channel] = affine_T(reduction[0]);
    dbeta[channel] = affine_T(reduction[1]);
  }
}

#define REGISTER_GROUP_NORM_BACKWARD_AFFINE(T, stat_T, affine_T, idx_T) \
  template [[host_name("group_norm_backward_affine_" #T "_" #stat_T     \
                       "_" #affine_T "_" #idx_T)]]                      \
  kernel void group_norm_backward_affine<T, stat_T, affine_T, idx_T>(   \
      device affine_T * dgamma [[buffer(0)]],                           \
      device affine_T * dbeta [[buffer(1)]],                            \
      device const T* dY [[buffer(2)]],                                 \
      device const T* X [[buffer(3)]],                                  \
      device const stat_T* mean [[buffer(4)]],                          \
      device const stat_T* rstd [[buffer(5)]],                          \
      constant GroupNormParams<idx_T>& params [[buffer(6)]],            \
      uint tgid [[threadgroup_position_in_grid]],                       \
      uint tid [[thread_position_in_threadgroup]],                      \
      uint tptg [[threads_per_threadgroup]]);

#define REGISTER_GROUP_NORM_BACKWARD_AFFINE_INDEX_TYPES(T, stat_T, affine_T) \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE(T, stat_T, affine_T, uint32_t);        \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE(T, stat_T, affine_T, uint64_t);

#define REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(T, stat_T)         \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE_INDEX_TYPES(T, stat_T, float); \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE_INDEX_TYPES(T, stat_T, half);  \
  REGISTER_GROUP_NORM_BACKWARD_AFFINE_INDEX_TYPES(T, stat_T, bfloat);

REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(float, float);
REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(half, float);
REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(bfloat, float);
REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(half, half);
REGISTER_GROUP_NORM_BACKWARD_AFFINE_TYPES(bfloat, bfloat);
