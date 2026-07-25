// Grouped GEMM kernels for mixture-of-experts expert projections.
//
// Rows of a packed 2D activation tensor are partitioned into contiguous
// per-expert groups by a cumulative offset tensor ``offs`` (int32, length E,
// ``offs[g]`` is the exclusive end row of group g). Each group multiplies by
// its own weight matrix.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {

constexpr int TILE = 16;

inline const __nv_bfloat16 *bf_in(const torch::Tensor &t) {
  return reinterpret_cast<const __nv_bfloat16 *>(t.data_ptr<at::BFloat16>());
}
inline __nv_bfloat16 *bf_out(torch::Tensor &t) {
  return reinterpret_cast<__nv_bfloat16 *>(t.data_ptr<at::BFloat16>());
}

// out[m, p] = sum_q A[m, q] * Wg[q, p] for the group g owning row m.
//
// FORWARD:  Q = K, P = N, contraction over K, out[m, n] = sum_k A[m,k] W[g,n,k]
// !FORWARD: Q = N, P = K, contraction over N, out[m, k] = sum_n A[m,n] W[g,n,k]
template <bool FORWARD>
__global__ void grouped_gemm_2d3d_kernel(const __nv_bfloat16 *__restrict__ A,
                                         const __nv_bfloat16 *__restrict__ W,
                                         __nv_bfloat16 *__restrict__ Out,
                                         const int32_t *__restrict__ offs, int N,
                                         int K) {
  const int Q = FORWARD ? K : N;
  const int P = FORWARD ? N : K;

  const int g = blockIdx.z;
  const int group_start = (g == 0) ? 0 : offs[g - 1];
  const int group_end = offs[g];

  const int tile_row0 = group_start + blockIdx.x * TILE;
  if (tile_row0 >= group_end) return;

  const int col0 = blockIdx.y * TILE;
  const int tx = threadIdx.x;  // column within tile
  const int ty = threadIdx.y;  // row within tile
  const int row = tile_row0 + ty;
  const int col = col0 + tx;

  __shared__ float As[TILE][TILE];  // [row][contraction]
  __shared__ float Bs[TILE][TILE];  // [contraction][column]

  const long wbase = (long)g * N * K;
  float acc = 0.0f;

  for (int q0 = 0; q0 < Q; q0 += TILE) {
    const int a_q = q0 + tx;
    As[ty][tx] = (row < group_end && a_q < Q)
                     ? __bfloat162float(A[(long)row * Q + a_q])
                     : 0.0f;

    const int b_q = q0 + ty;    // contraction index
    const int b_p = col0 + tx;  // output column
    float bval = 0.0f;
    if (b_q < Q && b_p < P) {
      const long widx = FORWARD ? wbase + (long)b_p * K + b_q
                                : wbase + (long)b_q * K + b_p;
      bval = __bfloat162float(W[widx]);
    }
    Bs[ty][tx] = bval;

    __syncthreads();
#pragma unroll
    for (int t = 0; t < TILE; ++t) acc += As[ty][t] * Bs[t][tx];
    __syncthreads();
  }

  if (row < group_end && col < P) Out[(long)row * P + col] = __float2bfloat16(acc);
}

// dW[g, n, k] = sum_{m in group g} DY[m, n] * X[m, k]
__global__ void grouped_gemm_wgrad_kernel(const __nv_bfloat16 *__restrict__ DY,
                                          const __nv_bfloat16 *__restrict__ X,
                                          __nv_bfloat16 *__restrict__ DW,
                                          const int32_t *__restrict__ offs,
                                          int N, int K) {
  const int g = blockIdx.z;
  const int group_start = (g == 0) ? 0 : offs[g - 1];
  const int group_end = offs[g];

  const int tx = threadIdx.x;  // k within tile
  const int ty = threadIdx.y;  // n within tile
  const int n = blockIdx.y * TILE + ty;
  const int k = blockIdx.x * TILE + tx;

  __shared__ float dyS[TILE][TILE];  // [row][n]
  __shared__ float xS[TILE][TILE];   // [row][k]

  float acc = 0.0f;
  for (int m0 = group_start; m0 < group_end; m0 += TILE) {
    const int m = m0 + ty;
    const int n_load = blockIdx.y * TILE + tx;
    dyS[ty][tx] = (m < group_end && n_load < N)
                      ? __bfloat162float(DY[(long)m * N + n_load])
                      : 0.0f;
    const int k_load = blockIdx.x * TILE + tx;
    xS[ty][tx] = (m < group_end && k_load < K)
                     ? __bfloat162float(X[(long)m * K + k_load])
                     : 0.0f;

    __syncthreads();
#pragma unroll
    for (int a = 0; a < TILE; ++a) acc += dyS[a][ty] * xS[a][tx];
    __syncthreads();
  }

  if (n < N && k < K) DW[((long)g * N + n) * K + k] = __float2bfloat16(acc);
}

torch::Tensor grouped_gemm_forward(torch::Tensor input, torch::Tensor weight,
                                   torch::Tensor offs) {
  input = input.contiguous();
  weight = weight.contiguous();
  offs = offs.contiguous();
  const int M = input.size(0);
  const int K = input.size(1);
  const int E = weight.size(0);
  const int N = weight.size(1);

  auto out = torch::empty({M, N}, input.options());
  const dim3 block(TILE, TILE);
  const dim3 grid((M + TILE - 1) / TILE, (N + TILE - 1) / TILE, E);
  grouped_gemm_2d3d_kernel<true><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      bf_in(input), bf_in(weight), bf_out(out), offs.data_ptr<int32_t>(), N, K);
  return out;
}

torch::Tensor grouped_gemm_dgrad(torch::Tensor grad_output, torch::Tensor weight,
                                 torch::Tensor offs) {
  grad_output = grad_output.contiguous();
  weight = weight.contiguous();
  offs = offs.contiguous();
  const int M = grad_output.size(0);
  const int E = weight.size(0);
  const int N = weight.size(1);
  const int K = weight.size(2);

  auto out = torch::empty({M, K}, grad_output.options());
  const dim3 block(TILE, TILE);
  const dim3 grid((M + TILE - 1) / TILE, (K + TILE - 1) / TILE, E);
  grouped_gemm_2d3d_kernel<false><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      bf_in(grad_output), bf_in(weight), bf_out(out), offs.data_ptr<int32_t>(), N,
      K);
  return out;
}

torch::Tensor grouped_gemm_wgrad(torch::Tensor grad_output, torch::Tensor input,
                                 torch::Tensor offs) {
  grad_output = grad_output.contiguous();
  input = input.contiguous();
  offs = offs.contiguous();
  const int N = grad_output.size(1);
  const int K = input.size(1);
  const int E = offs.size(0);

  auto out = torch::empty({E, N, K}, grad_output.options());
  const dim3 block(TILE, TILE);
  const dim3 grid((K + TILE - 1) / TILE, (N + TILE - 1) / TILE, E);
  grouped_gemm_wgrad_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      bf_in(grad_output), bf_in(input), bf_out(out), offs.data_ptr<int32_t>(), N,
      K);
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grouped_gemm_forward", &grouped_gemm_forward,
        "Grouped GEMM forward (2D x 3D, jagged on rows)");
  m.def("grouped_gemm_dgrad", &grouped_gemm_dgrad,
        "Grouped GEMM input gradient (2D x 3D, jagged on rows)");
  m.def("grouped_gemm_wgrad", &grouped_gemm_wgrad,
        "Grouped GEMM weight gradient (jagged on contraction rows)");
}
