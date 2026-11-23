#include <ATen/native/mps/kernels/Rnn.h>
#include <metal_stdlib>

using namespace metal;

template <typename T, typename idx_t>
[[max_total_threads_per_threadgroup(kCellMaxThreadsPerThreadgroup)]]
kernel void lstm_cell_forward(
    device T* gates,
    device const T* recurrent,
    device const T* cx,
    device T* cy,
    device T* hy,
    device const T* bias,
    constant uint& hidden_size,
    uint2 pos [[thread_position_in_grid]]) {
  const idx_t h = hidden_size, col = pos.x, row = pos.y;
  const idx_t idx = row * h + col, offset = row * kLstmGates * h + col;
  float4 g;
  for (uint k = 0; k < kLstmGates; ++k) {
    g[k] = float(gates[offset + k * h]) + float(recurrent[offset + k * h]) +
        float(bias[col + k * h]);
  }
  const float cell = precise::tanh(g.z);
  g = 1.0f / (1.0f + precise::exp(-g));
  g.z = cell;
  const float c = g.y * float(cx[idx]) + g.x * g.z;
  cy[idx] = T(c);
  hy[idx] = T(g.w * precise::tanh(c));
  for (uint k = 0; k < kLstmGates; ++k) {
    gates[offset + k * h] = T(g[k]);
  }
}

template <typename T, typename idx_t>
kernel void lstm_cell_backward(
    device const T* gates,
    device const T* cx,
    device const T* cy,
    device const T* dy,
    device const T* dh,
    device T* dc,
    device T* dg,
    constant uint& hidden_size,
    uint2 pos [[thread_position_in_grid]]) {
  const idx_t h = hidden_size, col = pos.x, row = pos.y;
  const idx_t idx = row * h + col, offset = row * kLstmGates * h + col;
  float4 g;
  for (uint k = 0; k < kLstmGates; ++k) {
    g[k] = float(gates[offset + k * h]);
  }
  const float grad_h = float(dy[idx]) + float(dh[idx]);
  const float c = precise::tanh(float(cy[idx]));
  const float grad_c = float(dc[idx]) + grad_h * g.w * (1.0f - c * c);
  dc[idx] = T(grad_c * g.y);
  const float4 grad =
      float4(grad_c * g.z, grad_c * float(cx[idx]), grad_c * g.x, grad_h * c) *
      float4(
          g.x * (1.0f - g.x),
          g.y * (1.0f - g.y),
          1.0f - g.z * g.z,
          g.w * (1.0f - g.w));
  for (uint k = 0; k < kLstmGates; ++k) {
    dg[offset + k * h] = T(grad[k]);
  }
}

#define REGISTER_LSTM(T)                                             \
  template [[host_name("lstm_cell_forward_" #T "_u32")]] [[kernel]]  \
  decltype(lstm_cell_forward<T, uint>) lstm_cell_forward<T, uint>;   \
  template [[host_name("lstm_cell_forward_" #T "_u64")]] [[kernel]]  \
  decltype(lstm_cell_forward<T, ulong>) lstm_cell_forward<T, ulong>; \
  template [[host_name("lstm_cell_backward_" #T "_u32")]] [[kernel]] \
  decltype(lstm_cell_backward<T, uint>) lstm_cell_backward<T, uint>; \
  template [[host_name("lstm_cell_backward_" #T "_u64")]] [[kernel]] \
  decltype(lstm_cell_backward<T, ulong>) lstm_cell_backward<T, ulong>

REGISTER_LSTM(float);
REGISTER_LSTM(half);
REGISTER_LSTM(bfloat);

template <typename T, typename idx_t>
[[max_total_threads_per_threadgroup(kCellMaxThreadsPerThreadgroup)]]
kernel void gru_cell_forward(
    device T* workspace,
    device const T* recurrent,
    device const T* hx,
    device T* hy,
    device const T* input_bias,
    device const T* hidden_bias,
    constant uint& hidden_size,
    uint2 pos [[thread_position_in_grid]]) {
  const idx_t h = hidden_size, col = pos.x, row = pos.y;
  const idx_t idx = row * h + col, offset = row * kGruWorkspace * h + col,
              gate_offset = row * kGruGates * h + col;
  float3 x, y;
  for (uint k = 0; k < kGruGates; ++k) {
    x[k] = float(workspace[offset + k * h]) + float(input_bias[col + k * h]);
    y[k] =
        float(recurrent[gate_offset + k * h]) + float(hidden_bias[col + k * h]);
  }
  const float2 rz = 1.0f / (1.0f + precise::exp(-(x.xy + y.xy)));
  const float n = precise::tanh(x.z + rz.x * y.z);
  hy[idx] = T(n + rz.y * (float(hx[idx]) - n));
  const float4 saved(rz, n, y.z);
  for (uint k = 0; k < kGruWorkspace; ++k) {
    workspace[offset + k * h] = T(saved[k]);
  }
}

// dh carries the gradient that reaches h through z and dh_recurrent the one
// through W_hh, both from the following time step.
template <typename T, typename idx_t>
kernel void gru_cell_backward(
    device const T* workspace,
    device const T* hx,
    device const T* dy,
    device const T* dh_recurrent,
    device T* dh,
    device T* input_grad,
    device T* hidden_grad,
    constant uint& hidden_size,
    uint2 pos [[thread_position_in_grid]]) {
  const idx_t h = hidden_size, col = pos.x, row = pos.y;
  const idx_t idx = row * h + col, offset = row * kGruWorkspace * h + col,
              gate_offset = row * kGruGates * h + col;
  float4 g;
  for (uint k = 0; k < kGruWorkspace; ++k) {
    g[k] = float(workspace[offset + k * h]);
  }
  const float grad_h =
      float(dy[idx]) + float(dh[idx]) + float(dh_recurrent[idx]);
  dh[idx] = T(grad_h * g.y);
  const float grad_n = grad_h * (1.0f - g.y) * (1.0f - g.z * g.z);
  float3 grad(
      grad_n * g.w * g.x * (1.0f - g.x),
      grad_h * (float(hx[idx]) - g.z) * g.y * (1.0f - g.y),
      grad_n);
  for (uint k = 0; k < kGruGates; ++k) {
    input_grad[gate_offset + k * h] = T(grad[k]);
  }
  grad.z *= g.x;
  for (uint k = 0; k < kGruGates; ++k) {
    hidden_grad[gate_offset + k * h] = T(grad[k]);
  }
}

#define REGISTER_GRU(T)                                             \
  template [[host_name("gru_cell_forward_" #T "_u32")]] [[kernel]]  \
  decltype(gru_cell_forward<T, uint>) gru_cell_forward<T, uint>;    \
  template [[host_name("gru_cell_forward_" #T "_u64")]] [[kernel]]  \
  decltype(gru_cell_forward<T, ulong>) gru_cell_forward<T, ulong>;  \
  template [[host_name("gru_cell_backward_" #T "_u32")]] [[kernel]] \
  decltype(gru_cell_backward<T, uint>) gru_cell_backward<T, uint>;  \
  template [[host_name("gru_cell_backward_" #T "_u64")]] [[kernel]] \
  decltype(gru_cell_backward<T, ulong>) gru_cell_backward<T, ulong>

REGISTER_GRU(float);
REGISTER_GRU(half);
REGISTER_GRU(bfloat);
