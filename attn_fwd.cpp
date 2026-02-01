// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <chrono>
#include <random>
#include <iomanip>
#include <map>

// clang-format off
// /opt/rocm/llvm/bin/clang++ -O3 -x hip --offload-arch=gfx950 -o attn_fwd attn_fwd.cpp && ./attn_fwd
// clang-format on

#define HIP_CHECK(call)                                                                    \
    do                                                                                     \
    {                                                                                      \
        hipError_t err = call;                                                             \
        if(err != hipSuccess)                                                              \
        {                                                                                  \
            printf("HIP error %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(err)); \
            exit(1);                                                                       \
        }                                                                                  \
    } while(0)

enum class CausalMaskType
{
    DISABLE      = 0,
    TOP_LEFT     = 1,
    BOTTOM_RIGHT = 2
};

std::map<CausalMaskType, std::string> CausalMaskTypeName = {
    {CausalMaskType::DISABLE, "DISABLE"},
    {CausalMaskType::TOP_LEFT, "TOP_LEFT"},
    {CausalMaskType::BOTTOM_RIGHT, "BOTTOM_RIGHT"}};

template <int BS,
          int HEAD_NUM,
          int SEQ_Q,
          int SEQ_KV,
          int HEAD_DIM,
          int STEP2_BLOCK_SIZE     = 256,
          bool ENABLE_DROPOUT_MASK = true,
          CausalMaskType MAKS_TYPE = CausalMaskType::DISABLE>
struct FmhaKernelConfig
{
    static constexpr int bs                        = BS;
    static constexpr int head_num                  = HEAD_NUM;
    static constexpr int seq_q                     = SEQ_Q;
    static constexpr int seq_kv                    = SEQ_KV;
    static constexpr int head_dim                  = HEAD_DIM;
    static constexpr int step2_block_size          = STEP2_BLOCK_SIZE;
    static constexpr bool enable_dropout_mask      = ENABLE_DROPOUT_MASK;
    static constexpr enum CausalMaskType mask_type = MAKS_TYPE;
};

template <typename T, typename Config>
__global__ void compute_scores_kernel(const T* Q, const T* K, T* scores, float scale)
{
    constexpr int seq_q                       = Config::seq_q;
    constexpr int seq_kv                      = Config::seq_kv;
    constexpr int head_dim                    = Config::head_dim;
    constexpr int warp_size                   = 64;
    constexpr int process_head_dim_per_thread = head_dim / warp_size;

    const uint32_t block_id  = blockIdx.x;
    const uint32_t thread_id = threadIdx.x;

    const T* Q_ptr = Q + block_id * seq_q * head_dim;
    const T* K_ptr = K + block_id * seq_kv * head_dim;
    T* scores_ptr  = scores + block_id * seq_q * seq_kv;

    // Prefetch K to shared memory
    __shared__ T fetch_K[seq_kv * head_dim];
#pragma unroll
    for(int i = 0; i < seq_kv; i++)
    {
#pragma unroll
        for(int k = 0; k < process_head_dim_per_thread; k++)
        {
            fetch_K[i * head_dim + thread_id * process_head_dim_per_thread + k] =
                K_ptr[i * head_dim + thread_id * process_head_dim_per_thread + k];
        }
    }
    __syncthreads();

    // compute scores = Q @ K^T / scale
    // Each thread computes partial sum over head_dim, then reduce
    __shared__ T reduce_buffer[head_dim / process_head_dim_per_thread];

#pragma unroll
    for(int i = 0; i < seq_q; i++)
    {
#pragma unroll
        for(int j = 0; j < seq_kv; j++)
        {
            // Each thread computes one element of the dot product
            T partial_sum = T(0.0f);
#pragma unroll
            for(int k = 0; k < process_head_dim_per_thread; k++)
            {
                partial_sum += Q_ptr[i * head_dim + thread_id * process_head_dim_per_thread + k] *
                               fetch_K[j * head_dim + thread_id * process_head_dim_per_thread + k];
            }
            reduce_buffer[thread_id] = partial_sum;
            __syncthreads();

            // Parallel reduction in shared memory
            for(int stride = head_dim / (2 * process_head_dim_per_thread); stride > 0; stride >>= 1)
            {
                if(thread_id < stride)
                {
                    reduce_buffer[thread_id] += reduce_buffer[thread_id + stride];
                }
                __syncthreads();
            }

            // Only thread 0 writes the final result
            if(thread_id == 0)
            {
                scores_ptr[i * seq_kv + j] = reduce_buffer[0] * scale;
            }
            __syncthreads();
        }
    }
}

template <typename T, typename Config>
__global__ void apply_mask_and_softmax_kernel(T* scores, const T* dropout_mask, float dropout_scale)
{
    const uint32_t block_id          = blockIdx.x;
    const uint32_t thread_id         = threadIdx.x;
    constexpr int seq_q              = Config::seq_q;
    constexpr int seq_kv             = Config::seq_kv;
    constexpr int block_size         = Config::step2_block_size;
    constexpr int per_score_size     = seq_q * seq_kv;
    constexpr int valid_thread_range = block_size / per_score_size * per_score_size;
    const uint32_t cur_block_offset  = block_id * valid_thread_range + thread_id;
    constexpr uint32_t total_elt     = Config::bs * Config::head_num * seq_q * seq_kv;
    bool is_tail                     = block_id * valid_thread_range + block_size >= total_elt;
    int real_row_num = is_tail ? (total_elt - block_id * valid_thread_range) / seq_kv
                               : valid_thread_range / seq_kv;

    if(cur_block_offset < total_elt && thread_id < valid_thread_range)
    {
        __shared__ T tmp_scores[valid_thread_range];
        constexpr int row_num = valid_thread_range / seq_kv;
        __shared__ T row_max[row_num];
        __shared__ T row_sum[row_num];

        T score_value         = scores[cur_block_offset];
        tmp_scores[thread_id] = score_value;

        // Apply causal mask before softmax
        if constexpr(Config::mask_type == CausalMaskType::TOP_LEFT)
        {
            int q_idx = (cur_block_offset % (seq_q * seq_kv)) / seq_kv;
            int k_idx = (cur_block_offset % (seq_q * seq_kv)) % seq_kv;
            if(k_idx > q_idx)
            {
                tmp_scores[thread_id] = T(-1e9f);
            }
        }
        else if constexpr(Config::mask_type == CausalMaskType::BOTTOM_RIGHT)
        {
            int q_idx = (cur_block_offset % (seq_q * seq_kv)) / seq_kv;
            int k_idx = (cur_block_offset % (seq_q * seq_kv)) % seq_kv;
            if(k_idx < q_idx)
            {
                tmp_scores[thread_id] = T(-1e9f);
            }
        }
        __syncthreads();

        // Find max for each row (numerically stable softmax)
        if(thread_id < real_row_num)
        {
            T max_val = T(-1e9f);
#pragma unroll
            for(int i = 0; i < seq_kv; i++)
            {
                max_val = max(max_val, tmp_scores[thread_id * seq_kv + i]);
            }
            row_max[thread_id] = max_val;
        }
        __syncthreads();

        // Compute exp(score - max) and sum for each row
        T exp_val             = T(exp(float(tmp_scores[thread_id] - row_max[thread_id / seq_kv])));
        tmp_scores[thread_id] = exp_val;
        __syncthreads();

        if(thread_id < real_row_num)
        {
            T sum = T(0.0f);
#pragma unroll
            for(int i = 0; i < seq_kv; i++)
            {
                sum += tmp_scores[thread_id * seq_kv + i];
            }
            row_sum[thread_id] = sum;
        }
        __syncthreads();

        // Normalize and apply dropout
        T attn_weight = tmp_scores[thread_id] / row_sum[thread_id / seq_kv];

        if constexpr(Config::enable_dropout_mask)
        {
            attn_weight = attn_weight * dropout_mask[cur_block_offset] * dropout_scale;
        }

        scores[cur_block_offset] = attn_weight;
    }
}

template <typename T, typename Config>
__global__ void compute_output_kernel(const T* attn_weights, const T* V, T* O)
{
    constexpr int seq_q                       = Config::seq_q;
    constexpr int seq_kv                      = Config::seq_kv;
    constexpr int head_dim                    = Config::head_dim;
    constexpr int warp_size                   = 64;
    constexpr int process_head_dim_per_thread = head_dim / warp_size;

    const uint32_t block_id  = blockIdx.x;
    const uint32_t thread_id = threadIdx.x;

    const T* attn_weights_ptr = attn_weights + block_id * seq_q * seq_kv;
    const T* V_ptr            = V + block_id * seq_kv * head_dim;
    T* O_ptr                  = O + block_id * seq_q * head_dim;

    // Prefetch V to shared memory
    __shared__ T fetch_V[seq_kv * head_dim];
#pragma unroll
    for(int i = 0; i < seq_kv; i++)
    {
#pragma unroll
        for(int k = 0; k < process_head_dim_per_thread; k++)
        {
            fetch_V[i * head_dim + thread_id * process_head_dim_per_thread + k] =
                V_ptr[i * head_dim + thread_id * process_head_dim_per_thread + k];
        }
    }
    __syncthreads();

    // compute O = attn_weights @ V
#pragma unroll
    for(int i = 0; i < seq_q; i++)
    {
        T sums[process_head_dim_per_thread];
#pragma unroll
        for(int k = 0; k < process_head_dim_per_thread; k++)
        {
            sums[k] = T(0.0f);
        }
#pragma unroll
        for(int j = 0; j < seq_kv; j++)
        {
#pragma unroll
            for(int k = 0; k < process_head_dim_per_thread; k++)
            {
                sums[k] += attn_weights_ptr[i * seq_kv + j] *
                           fetch_V[j * head_dim + thread_id * process_head_dim_per_thread + k];
            }
        }
#pragma unroll
        for(int k = 0; k < process_head_dim_per_thread; k++)
        {
            O_ptr[i * head_dim + thread_id * process_head_dim_per_thread + k] = sums[k];
        }
    }
}

template <typename T, typename Config>
struct AttnForwardKernelLauncher
{

    static size_t calc_workspace_size()
    {
        constexpr int bs       = Config::bs;
        constexpr int head_num = Config::head_num;
        constexpr int seq_q    = Config::seq_q;
        constexpr int seq_kv   = Config::seq_kv;

        size_t workspace_size = bs * head_num * seq_q * seq_kv * sizeof(T);
        return workspace_size;
    }

    static void run_attn_fwd_kernel(const T* Q,
                                    const T* K,
                                    const T* V,
                                    const T* dropout_mask,
                                    float dropout_p,
                                    float sqr_dk_scale,
                                    T* O,
                                    T* attn_weights,
                                    T* workspace)
    {
        constexpr int bs        = Config::bs;
        constexpr int head_num  = Config::head_num;
        constexpr int seq_q     = Config::seq_q;
        constexpr int seq_kv    = Config::seq_kv;
        constexpr int head_dim  = Config::head_dim;
        constexpr int warp_size = 64;

        constexpr int merge_bs = bs * head_num;
        float scale            = sqr_dk_scale;
        float dropout_scale    = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;

        dim3 grid(merge_bs);
        dim3 block(warp_size);

        // Step 1: Compute scores = Q @ K^T / sqrt(d_k)
        compute_scores_kernel<T, Config><<<grid, block>>>(Q, K, workspace, scale);

        // Step 2: Apply mask and softmax (with dropout)
        // constexpr int work_thread_num =
        //     Config::step2_block_size / (seq_q * seq_kv) * (seq_q * seq_kv);
        // dim3 grid2((merge_bs * seq_q * seq_kv + work_thread_num - 1) / work_thread_num);
        // dim3 block2(Config::step2_block_size);
        // apply_mask_and_softmax_kernel<T, Config>
        //     <<<grid2, block2>>>(workspace, dropout_mask, dropout_scale);

        // Copy attention weights to output if needed
        if(attn_weights != nullptr)
        {
            HIP_CHECK(hipMemcpy(attn_weights,
                                workspace,
                                merge_bs * seq_q * seq_kv * sizeof(T),
                                hipMemcpyDeviceToDevice));
        }

        // Step 3: Compute output = attn_weights @ V
        // compute_output_kernel<T, Config><<<grid, block>>>(workspace, V, O);
    }
};

// Helper function: Matrix multiplication C = A @ B
// A: [rows_a, cols_a], B: [cols_a, cols_b], C: [rows_a, cols_b]
template <typename T>
void matmul(const T* A, const T* B, T* C, int rows_a, int cols_a, int cols_b)
{
    for(int i = 0; i < rows_a; i++)
    {
        for(int j = 0; j < cols_b; j++)
        {
            float sum = 0.0f;
            for(int k = 0; k < cols_a; k++)
            {
                sum += float(A[i * cols_a + k]) * float(B[k * cols_b + j]);
            }
            C[i * cols_b + j] = T(sum);
        }
    }
}

// Helper function: Matrix transpose
template <typename T>
void transpose(const T* A, T* A_T, int rows, int cols)
{
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            A_T[j * rows + i] = A[i * cols + j];
        }
    }
}

/**
 * Multi-Head Attention Forward Pass (CPU Reference Implementation)
 */
template <typename T>
void attn_forward(const T* Q,
                  const T* K,
                  const T* V,
                  const T* dropout_mask,
                  float dropout_p,
                  T* O,
                  T* attn_weights,
                  int batch,
                  int head_num,
                  int q_seq,
                  int kv_seq,
                  int head_dim,
                  CausalMaskType mask_type)
{

    float scale         = 1.0f / std::sqrt(static_cast<float>(head_dim));
    float dropout_scale = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;

    // Allocate temporary buffers
    std::vector<T> K_T(head_dim * kv_seq);
    std::vector<T> scores(q_seq * kv_seq);
    std::vector<T> attn_probs(q_seq * kv_seq);

    // Initialize output to zero
    std::memset(O, 0, batch * head_num * q_seq * head_dim * sizeof(T));
    if(attn_weights != nullptr)
    {
        std::memset(attn_weights, 0, batch * head_num * q_seq * kv_seq * sizeof(T));
    }

    // Process each batch and head
    for(int b = 0; b < batch; b++)
    {
        for(int h = 0; h < head_num; h++)
        {
            int offset_Q       = (b * head_num + h) * q_seq * head_dim;
            int offset_K       = (b * head_num + h) * kv_seq * head_dim;
            int offset_V       = (b * head_num + h) * kv_seq * head_dim;
            int offset_O       = (b * head_num + h) * q_seq * head_dim;
            int offset_attn    = (b * head_num + h) * q_seq * kv_seq;
            int offset_dropout = dropout_mask ? (b * head_num + h) * q_seq * kv_seq : 0;

            const T* Q_bh       = Q + offset_Q;
            const T* K_bh       = K + offset_K;
            const T* V_bh       = V + offset_V;
            const T* dropout_bh = dropout_mask ? dropout_mask + offset_dropout : nullptr;

            T* O_bh    = O + offset_O;
            T* attn_bh = attn_weights ? attn_weights + offset_attn : nullptr;

            // Step 1: Compute scores = Q @ K^T / sqrt(d_k)
            // Q: [q_seq, head_dim], K: [kv_seq, head_dim] -> scores: [q_seq, kv_seq]
            transpose(K_bh, K_T.data(), kv_seq, head_dim);
            matmul(Q_bh, K_T.data(), scores.data(), q_seq, head_dim, kv_seq);

            for(int i = 0; i < q_seq * kv_seq; i++)
            {
                scores[i] = T(float(scores[i]) * scale);
            }

            // Step 2: Apply causal mask
            if(mask_type == CausalMaskType::TOP_LEFT)
            {
                for(int i = 0; i < q_seq; i++)
                {
                    for(int j = 0; j < kv_seq; j++)
                    {
                        if(j > i)
                        {
                            scores[i * kv_seq + j] = T(-1e9f);
                        }
                    }
                }
            }
            else if(mask_type == CausalMaskType::BOTTOM_RIGHT)
            {
                for(int i = 0; i < q_seq; i++)
                {
                    for(int j = 0; j < kv_seq; j++)
                    {
                        if(j < i)
                        {
                            scores[i * kv_seq + j] = T(-1e9f);
                        }
                    }
                }
            }

            // Step 3: Softmax
            for(int i = 0; i < q_seq; i++)
            {
                // Find max for numerical stability
                float max_val = -1e9f;
                for(int j = 0; j < kv_seq; j++)
                {
                    max_val = std::max(max_val, float(scores[i * kv_seq + j]));
                }

                // Compute exp and sum
                float sum = 0.0f;
                for(int j = 0; j < kv_seq; j++)
                {
                    attn_probs[i * kv_seq + j] =
                        T(std::exp(float(scores[i * kv_seq + j]) - max_val));
                    sum += float(attn_probs[i * kv_seq + j]);
                }

                // Normalize
                for(int j = 0; j < kv_seq; j++)
                {
                    attn_probs[i * kv_seq + j] = T(float(attn_probs[i * kv_seq + j]) / sum);
                }
            }

            // Step 4: Apply dropout
            if(dropout_p > 0.0f && dropout_bh != nullptr)
            {
                for(int i = 0; i < q_seq * kv_seq; i++)
                {
                    attn_probs[i] = T(float(attn_probs[i]) * float(dropout_bh[i]) * dropout_scale);
                }
            }

            // Save attention weights if requested
            if(attn_bh != nullptr)
            {
                std::memcpy(attn_bh, attn_probs.data(), q_seq * kv_seq * sizeof(T));
            }

            // Step 5: Compute output = attn_probs @ V
            // attn_probs: [q_seq, kv_seq], V: [kv_seq, head_dim] -> O: [q_seq, head_dim]
            matmul(attn_probs.data(), V_bh, O_bh, q_seq, kv_seq, head_dim);
        }
    }
}

/**
 * Test run_attn_fwd_kernel correctness and bandwidth
 */
template <typename DataType, typename Config>
void test_run_attn_fwd_kernel(
    float dropout_p, int warmup_iters, int test_iters, bool check_correctness, bool dump_err)
{
    using Launcher = AttnForwardKernelLauncher<DataType, Config>;

    constexpr int bs       = Config::bs;
    constexpr int head_num = Config::head_num;
    constexpr int seq_q    = Config::seq_q;
    constexpr int seq_kv   = Config::seq_kv;
    constexpr int head_dim = Config::head_dim;

    // Calculate sizes
    size_t size_Q            = bs * head_num * seq_q * head_dim;
    size_t size_K            = bs * head_num * seq_kv * head_dim;
    size_t size_V            = bs * head_num * seq_kv * head_dim;
    size_t size_O            = bs * head_num * seq_q * head_dim;
    size_t size_attn_weights = bs * head_num * seq_q * seq_kv;
    size_t size_dropout_mask = bs * head_num * seq_q * seq_kv;

    // Allocate host memory
    std::vector<DataType> h_Q(size_Q);
    std::vector<DataType> h_K(size_K);
    std::vector<DataType> h_V(size_V);
    std::vector<DataType> h_dropout_mask(size_dropout_mask);
    std::vector<DataType> h_O_gpu(size_O);
    std::vector<DataType> h_O_cpu(size_O);
    std::vector<DataType> h_attn_weights_gpu(size_attn_weights);
    std::vector<DataType> h_attn_weights_cpu(size_attn_weights);

    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for(size_t i = 0; i < size_Q; i++)
        h_Q[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_K; i++)
        h_K[i] = DataType(dis(gen));
    for(size_t i = 0; i < size_V; i++)
        h_V[i] = DataType(dis(gen));

    // Initialize dropout mask (1 = keep, 0 = drop)
    for(size_t i = 0; i < size_dropout_mask; i++)
    {
        h_dropout_mask[i] = Config::enable_dropout_mask
                                ? DataType(dis(gen) > dropout_p ? 1.0f : 0.0f)
                                : DataType(1.0f);
    }

    // Compute CPU reference
    float sqr_dk_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if(check_correctness)
        attn_forward(h_Q.data(),
                     h_K.data(),
                     h_V.data(),
                     Config::enable_dropout_mask ? h_dropout_mask.data() : nullptr,
                     dropout_p,
                     h_O_cpu.data(),
                     h_attn_weights_cpu.data(),
                     bs,
                     head_num,
                     seq_q,
                     seq_kv,
                     head_dim,
                     Config::mask_type);

    // Allocate device memory
    DataType *d_Q, *d_K, *d_V;
    DataType* d_dropout_mask;
    DataType *d_O, *d_attn_weights, *d_workspace;

    HIP_CHECK(hipMalloc(&d_Q, size_Q * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_K, size_K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_V, size_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_dropout_mask, size_dropout_mask * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_O, size_O * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_attn_weights, size_attn_weights * sizeof(DataType)));

    size_t workspace_size = Launcher::calc_workspace_size();
    HIP_CHECK(hipMalloc(&d_workspace, workspace_size));

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), size_Q * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K.data(), size_K * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), size_V * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dropout_mask,
                        h_dropout_mask.data(),
                        size_dropout_mask * sizeof(DataType),
                        hipMemcpyHostToDevice));

    // Warmup runs
    for(int i = 0; i < warmup_iters; i++)
    {
        Launcher::run_attn_fwd_kernel(d_Q,
                                      d_K,
                                      d_V,
                                      Config::enable_dropout_mask ? d_dropout_mask : nullptr,
                                      dropout_p,
                                      sqr_dk_scale,
                                      d_O,
                                      d_attn_weights,
                                      d_workspace);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Timed runs
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    for(int i = 0; i < test_iters; i++)
    {
        Launcher::run_attn_fwd_kernel(d_Q,
                                      d_K,
                                      d_V,
                                      Config::enable_dropout_mask ? d_dropout_mask : nullptr,
                                      dropout_p,
                                      sqr_dk_scale,
                                      d_O,
                                      d_attn_weights,
                                      d_workspace);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    double avg_time_ms = elapsed_ms / test_iters;

    // Calculate TFLOPS
    // Main matrix operations in attention forward:
    // 1. scores = Q @ K^T: [q_seq, head_dim] @ [head_dim, kv_seq]
    // 2. O = attn_weights @ V: [q_seq, kv_seq] @ [kv_seq, head_dim]
    // Each matmul: FLOPs = 2 * M * N * K (multiply-add)
    double flops_per_batch_head = 2.0 * seq_q * seq_kv * head_dim + // scores
                                  2.0 * seq_q * head_dim * seq_kv;  // O
    double total_flops = flops_per_batch_head * bs * head_num;
    double tflops      = (total_flops / 1e12) / (avg_time_ms / 1000.0);

    // Copy results back
    HIP_CHECK(hipMemcpy(h_O_gpu.data(), d_O, size_O * sizeof(DataType), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_attn_weights_gpu.data(),
                        d_attn_weights,
                        size_attn_weights * sizeof(DataType),
                        hipMemcpyDeviceToHost));

    // Check correctness
    auto check_results = [&](const std::vector<DataType>& gpu,
                             const std::vector<DataType>& cpu,
                             const std::string& name,
                             float tolerance = 1e-1) {
        float max_diff     = 0.0f;
        float max_rel_diff = 0.0f;
        size_t diff_count  = 0;

        for(size_t i = 0; i < gpu.size(); i++)
        {
            float diff     = std::abs(float(gpu[i]) - float(cpu[i]));
            float rel_diff = diff / (std::abs(float(cpu[i])) + 1e-6f);
            max_diff       = std::max(max_diff, diff);
            max_rel_diff   = std::max(max_rel_diff, rel_diff);
            if(rel_diff > tolerance)
            {
                if(dump_err)
                    std::cout << name << " mismatch at index " << i << ": GPU=" << std::fixed
                              << std::setprecision(10) << static_cast<float>(gpu[i])
                              << ", CPU=" << std::fixed << std::setprecision(10)
                              << static_cast<float>(cpu[i]) << ", abs_diff=" << diff
                              << ", rel_diff=" << rel_diff << std::endl;
                diff_count++;
            }
        }

        std::cout << name << " check:" << std::endl;
        std::cout << "  Max absolute diff: " << max_diff << std::endl;
        std::cout << "  Max relative diff: " << max_rel_diff << std::endl;
        std::cout << "  Elements exceeding tolerance: " << diff_count << " / " << gpu.size()
                  << std::endl;
        std::cout << "  Status: " << (max_rel_diff < tolerance ? "PASS" : "FAIL") << std::endl;
    };

    // Calculate bandwidth
    size_t bytes_read = (size_Q + size_K + size_V) * sizeof(DataType);
    if(Config::enable_dropout_mask)
        bytes_read += size_dropout_mask * sizeof(DataType);

    size_t bytes_write    = size_O * sizeof(DataType);
    size_t total_bytes    = bytes_read + bytes_write;
    double bandwidth_gbps = (total_bytes / 1e9) / (avg_time_ms / 1000.0);

    // Print results
    std::cout << "\n===== run_attn_fwd_kernel Test =====" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << bs << std::endl;
    std::cout << "  Heads: " << head_num << std::endl;
    std::cout << "  Q sequence length: " << seq_q << std::endl;
    std::cout << "  KV sequence length: " << seq_kv << std::endl;
    std::cout << "  Head dimension: " << head_dim << std::endl;
    std::cout << "  Dropout: " << (Config::enable_dropout_mask ? "enabled" : "disabled")
              << std::endl;
    std::cout << "  Mask: " << (CausalMaskTypeName[Config::mask_type]) << std::endl;
    std::cout << std::endl;

    if(check_correctness)
    {
        std::cout << "Correctness:" << std::endl;
        check_results(h_O_gpu, h_O_cpu, "Output");
        check_results(h_attn_weights_gpu, h_attn_weights_cpu, "Attention Weights");
        std::cout << std::endl;
    }
    std::cout << "Memory:" << std::endl;
    std::cout << "  Total data read: " << std::fixed << std::setprecision(2) << bytes_read / 1e6
              << " MB" << std::endl;
    std::cout << "  Total data write: " << bytes_write / 1e6 << " MB" << std::endl;
    std::cout << "  Total data transfer: " << total_bytes / 1e6 << " MB" << std::endl;
    std::cout << "  Workspace size: " << workspace_size / 1e6 << " MB" << std::endl;
    std::cout << std::endl;

    std::cout << "Performance:" << std::endl;
    std::cout << "  Average time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms"
              << std::endl;
    std::cout << "  Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth_gbps << " GB/s"
              << std::endl;
    std::cout << "  TFLOPS: " << std::fixed << std::setprecision(2) << tflops << std::endl;
    std::cout << "====================================\n" << std::endl;

    // Cleanup
    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_K));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_dropout_mask));
    HIP_CHECK(hipFree(d_O));
    HIP_CHECK(hipFree(d_attn_weights));
    HIP_CHECK(hipFree(d_workspace));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

// Template metaprogramming: recursive template to iterate over SEQ_KV values
template <int SEQ_KV, int MAX_SEQ_KV>
struct TestRunner
{
    template <typename DataType,
              int BS,
              int HEAD_NUM,
              int SEQ_Q,
              int HEAD_DIM,
              int STEP2_BLOCK_SIZE,
              bool ENABLE_DROPOUT_MASK,
              CausalMaskType MASK_TYPE>
    static void
    run(float dropout_p, int warmup_iters, int test_iters, bool check_correctness, bool dump_err)
    {
        using KernelConfig = FmhaKernelConfig<BS,
                                              HEAD_NUM,
                                              SEQ_Q,
                                              SEQ_KV,
                                              HEAD_DIM,
                                              STEP2_BLOCK_SIZE,
                                              ENABLE_DROPOUT_MASK,
                                              MASK_TYPE>;
        test_run_attn_fwd_kernel<DataType, KernelConfig>(
            dropout_p, warmup_iters, test_iters, check_correctness, dump_err);

        // Recursive call for next SEQ_KV value
        TestRunner<SEQ_KV + 1, MAX_SEQ_KV>::template run<DataType,
                                                         BS,
                                                         HEAD_NUM,
                                                         SEQ_Q,
                                                         HEAD_DIM,
                                                         STEP2_BLOCK_SIZE,
                                                         ENABLE_DROPOUT_MASK,
                                                         MASK_TYPE>(
            dropout_p, warmup_iters, test_iters, check_correctness, dump_err);
    }
};

// Termination condition: when SEQ_KV reaches MAX_SEQ_KV
template <int MAX_SEQ_KV>
struct TestRunner<MAX_SEQ_KV, MAX_SEQ_KV>
{
    template <typename DataType,
              int BS,
              int HEAD_NUM,
              int SEQ_Q,
              int HEAD_DIM,
              int STEP2_BLOCK_SIZE,
              bool ENABLE_DROPOUT_MASK,
              CausalMaskType MASK_TYPE>
    static void
    run(float dropout_p, int warmup_iters, int test_iters, bool check_correctness, bool dump_err)
    {
        using KernelConfig = FmhaKernelConfig<BS,
                                              HEAD_NUM,
                                              SEQ_Q,
                                              MAX_SEQ_KV,
                                              HEAD_DIM,
                                              STEP2_BLOCK_SIZE,
                                              ENABLE_DROPOUT_MASK,
                                              MASK_TYPE>;
        test_run_attn_fwd_kernel<DataType, KernelConfig>(
            dropout_p, warmup_iters, test_iters, check_correctness, dump_err);
    }
};

int main(int argc, char const* argv[])
{
    std::cout << "\n========== Testing with bfloat16 (SEQ_KV from 4 to 16) ==========" << std::endl;

    // Using template metaprogramming to generate tests for SEQ_KV from 4 to 16
    // Template parameters: DataType, BS, HEAD_NUM, SEQ_Q, HEAD_DIM, STEP2_BLOCK_SIZE,
    // ENABLE_DROPOUT_MASK, MASK_TYPE
    TestRunner<4, 16>::run<hip_bfloat16, 30720, 32, 1, 128, 128, false, CausalMaskType::TOP_LEFT>(
        0,     // dropout_p
        0,     // warmup_iters
        1,     // test_iters
        false, // check_correctness
        false  // dump_err
    );

    return 0;
}
