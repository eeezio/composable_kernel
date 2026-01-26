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

// clang-format off
// /opt/rocm/llvm/bin/clang++ -O3 -x hip --save-temps --offload-arch=gfx950 -o test-f8f4 test-f8f4.cpp && ./test-f8f4
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

template <int BS,
          int HEAD_NUM,
          int SEQ_Q,
          int SEQ_KV,
          int HEAD_DIM,
          int STEP2_BLOCK_SIZE     = 256,
          bool ENABLE_DROPOUT_MASK = true,
          bool ENABLE_MASK         = true>
struct FmhaKernelConfig
{
    static constexpr int bs                   = BS;
    static constexpr int head_num             = HEAD_NUM;
    static constexpr int seq_q                = SEQ_Q;
    static constexpr int seq_kv               = SEQ_KV;
    static constexpr int head_dim             = HEAD_DIM;
    static constexpr int step2_block_size     = STEP2_BLOCK_SIZE;
    static constexpr bool enable_dropout_mask = ENABLE_DROPOUT_MASK;
    static constexpr bool enable_mask         = ENABLE_MASK;
};

template <typename T, typename Config>
__global__ void compute_grad_v_and_grad_attn_kernel(const T* attn_weights,
                                                    const T* grad_O,
                                                    const T* V,
                                                    T* grad_V,
                                                    T* workspace) // store grad_attn
{
    constexpr int seq_q                       = Config::seq_q;
    constexpr int seq_kv                      = Config::seq_kv;
    constexpr int head_dim                    = Config::head_dim;
    constexpr int warp_size                   = 64;
    constexpr int process_head_dim_per_thread = head_dim / warp_size;

    const uint32_t block_id  = blockIdx.x;
    const uint32_t thread_id = threadIdx.x;

    const T* attn_weights_ptr = attn_weights + block_id * seq_q * seq_kv;
    const T* grad_O_ptr       = grad_O + block_id * seq_q * head_dim;
    const T* V_ptr            = V + block_id * seq_kv * head_dim;
    T* grad_V_ptr             = grad_V + block_id * seq_kv * head_dim;
    T* workspace_ptr          = workspace + block_id * seq_q * seq_kv;
    __shared__ T fetch_grad_O[seq_q * head_dim];
// fetch grad_O to shared memory
#pragma unroll
    for(int i = 0; i < seq_q; i++)
    {
#pragma unroll
        for(int k = 0; k < process_head_dim_per_thread; k++)
        {
            fetch_grad_O[i * head_dim + thread_id * process_head_dim_per_thread + k] =
                grad_O_ptr[i * head_dim + thread_id * process_head_dim_per_thread + k];
        }
    }
    __syncthreads();
// compute grad_V = attn_weights^T @ grad_O
#pragma unroll
    for(int i = 0; i < seq_kv; i++)
    {
        T sums[process_head_dim_per_thread];
#pragma unroll
        for(int k = 0; k < process_head_dim_per_thread; k++)
        {
            sums[k] = T(0.0f);
        }
#pragma unroll
        for(int j = 0; j < seq_q; j++)
        {
#pragma unroll
            for(int k = 0; k < process_head_dim_per_thread; k++)
            {
                sums[k] += attn_weights_ptr[j * seq_kv + i] *
                           fetch_grad_O[j * head_dim + thread_id * process_head_dim_per_thread + k];
            }
        }
#pragma unroll
        for(int k = 0; k < process_head_dim_per_thread; k++)
        {
            grad_V_ptr[i * head_dim + thread_id * process_head_dim_per_thread + k] = sums[k];
        }
    }
    __syncthreads();

    // compute grad_attn = grad_O @ V^T
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
                partial_sum +=
                    fetch_grad_O[i * head_dim + thread_id * process_head_dim_per_thread + k] *
                    V_ptr[j * head_dim + thread_id * process_head_dim_per_thread + k];
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
                workspace_ptr[i * seq_kv + j] = reduce_buffer[0];
            }
            __syncthreads();
        }
    }
}

template <typename T, typename mask_type, typename Config>
__global__ void apply_softmax_backward_kernel(const T* attn_weights,
                                              const T* dropout_mask,
                                              const mask_type* mask,
                                              T* grad_attn,
                                              float dropout_scale)
{
    const uint32_t block_id        = blockIdx.x;
    const uint32_t thread_id       = threadIdx.x;
    constexpr int seq_q            = Config::seq_q;
    constexpr int seq_kv           = Config::seq_kv;
    constexpr int bs_num_per_block = Config::step2_block_size;
    static_assert(bs_num_per_block / (seq_kv * seq_q) > 0 &&
                      bs_num_per_block % (seq_q * seq_kv) == 0,
                  "wrong!");
    const uint32_t cur_block_offset = block_id * bs_num_per_block + thread_id;
    __shared__ T tmp_grad_score[bs_num_per_block];
    constexpr int reduce_score_num = bs_num_per_block / seq_kv;
    __shared__ T reduce_grad_score[reduce_score_num];
    constexpr int loop_num = bs_num_per_block / (seq_q * seq_kv);

    T grad_attn_value = grad_attn[cur_block_offset];
    if constexpr(Config::enable_dropout_mask)
    {
        grad_attn_value = grad_attn_value * dropout_mask[cur_block_offset] * dropout_scale;
    }
    T attn_weight             = attn_weights[cur_block_offset];
    T grad_score              = grad_attn_value * attn_weight;
    tmp_grad_score[thread_id] = grad_score;
    __syncthreads();
    // reduce within block
    if(thread_id < reduce_score_num)
    {
        T sum = T(0.0f);
#pragma unroll
        for(int i = 0; i < seq_kv; i++)
        {
            sum += tmp_grad_score[thread_id * seq_kv + i];
        }
        reduce_grad_score[thread_id] = sum;
    }
    __syncthreads();
    grad_score -= attn_weight * reduce_grad_score[thread_id / seq_kv];
    if constexpr(Config::enable_mask)
    {
        mask_type cur_thread_mask = mask[cur_block_offset];
        if(cur_thread_mask == 0)
        {
            grad_score = T(0.0f);
        }
    }
    grad_attn[cur_block_offset] = grad_score;
}

template <typename T, typename Config>
__global__ void compute_grad_q_and_grad_k_kernel(
    const T* grad_scores, const T* Q, const T* K, T* grad_Q, T* grad_K, float scale)
{
    const uint32_t block_id                   = blockIdx.x;
    const uint32_t thread_id                  = threadIdx.x;
    constexpr int seq_q                       = Config::seq_q;
    constexpr int seq_kv                      = Config::seq_kv;
    constexpr int head_dim                    = Config::head_dim;
    constexpr int warp_size                   = 64;
    constexpr int process_head_dim_per_thread = head_dim / warp_size;

    const T* grad_scores_ptr = grad_scores + block_id * seq_q * seq_kv;
    const T* Q_ptr           = Q + block_id * seq_q * head_dim;
    const T* K_ptr           = K + block_id * seq_kv * head_dim;
    T* grad_Q_ptr            = grad_Q + block_id * seq_q * head_dim;
    T* grad_K_ptr            = grad_K + block_id * seq_kv * head_dim;

    // compute grad_Q = grad_scores @ K
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
                sums[k] += grad_scores_ptr[i * seq_kv + j] *
                           K_ptr[j * head_dim + thread_id * process_head_dim_per_thread + k];
            }
        }
#pragma unroll
        for(int k = 0; k < process_head_dim_per_thread; k++)
        {
            grad_Q_ptr[i * head_dim + thread_id * process_head_dim_per_thread + k] =
                sums[k] * scale;
        }
    }
    // compute grad_K = grad_scores^T @ Q
#pragma unroll
    for(int i = 0; i < seq_kv; i++)
    {
        T sums[process_head_dim_per_thread];
#pragma unroll
        for(int k = 0; k < process_head_dim_per_thread; k++)
        {
            sums[k] = T(0.0f);
        }
#pragma unroll
        for(int j = 0; j < seq_q; j++)
        {
#pragma unroll
            for(int k = 0; k < process_head_dim_per_thread; k++)
            {
                sums[k] += grad_scores_ptr[j * seq_kv + i] *
                           Q_ptr[j * head_dim + thread_id * process_head_dim_per_thread + k];
            }
        }
#pragma unroll
        for(int k = 0; k < process_head_dim_per_thread; k++)
        {
            grad_K_ptr[i * head_dim + thread_id * process_head_dim_per_thread + k] =
                sums[k] * scale;
        }
    }
}

template <typename T, typename Config>
struct AttnBackwardKernelLauncher
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

    static void run_attn_bwd_kernel(const T* Q,
                                    const T* K,
                                    const T* V,
                                    const T* grad_O,
                                    const T* attn_weights,
                                    const uint8_t* mask,
                                    const T* dropout_mask,
                                    float dropout_p,
                                    float sqr_dk_scale,
                                    T* grad_Q,
                                    T* grad_K,
                                    T* grad_V,
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

        compute_grad_v_and_grad_attn_kernel<T, Config>
            <<<grid, block>>>(attn_weights, grad_O, V, grad_V, workspace);

        static_assert(merge_bs * seq_q * seq_kv / Config::step2_block_size > 0);
        dim3 grid2(merge_bs * seq_q * seq_kv / Config::step2_block_size);
        dim3 block2(Config::step2_block_size);
        apply_softmax_backward_kernel<T, uint8_t, Config>
            <<<grid2, block2>>>(attn_weights, dropout_mask, mask, workspace, dropout_scale);

        compute_grad_q_and_grad_k_kernel<T, Config>
            <<<grid, block>>>(workspace, Q, K, grad_Q, grad_K, scale);
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

// Helper function: Sum along last dimension
template <typename T>
void sum_last_dim(const T* A, T* sums, int rows, int cols)
{
    for(int i = 0; i < rows; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < cols; j++)
        {
            sum += float(A[i * cols + j]);
        }
        sums[i] = T(sum);
    }
}

/**
 * Multi-Head Attention Backward Pass (CPU Reference Implementation)
 */
template <typename T>
void attn_backward(const T* Q,
                   const T* K,
                   const T* V,
                   const T* grad_O,
                   const T* attn_weights,
                   const uint8_t* mask,
                   const T* dropout_mask,
                   float dropout_p,
                   T* grad_Q,
                   T* grad_K,
                   T* grad_V,
                   int batch,
                   int head_num,
                   int q_seq,
                   int kv_seq,
                   int head_dim)
{

    float scale         = 1.0f / std::sqrt(static_cast<float>(head_dim));
    float dropout_scale = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;

    // Allocate temporary buffers
    std::vector<T> V_T(kv_seq * head_dim);
    std::vector<T> grad_attn(q_seq * kv_seq);
    std::vector<T> grad_scores(q_seq * kv_seq);
    std::vector<T> attn_T(kv_seq * q_seq);
    std::vector<T> grad_scores_T(kv_seq * q_seq);
    std::vector<T> row_sums(q_seq);
    std::vector<T> K_T(head_dim * kv_seq);
    std::vector<T> Q_T(head_dim * q_seq);

    // Initialize gradients to zero
    std::memset(grad_Q, 0, batch * head_num * q_seq * head_dim * sizeof(T));
    std::memset(grad_K, 0, batch * head_num * kv_seq * head_dim * sizeof(T));
    std::memset(grad_V, 0, batch * head_num * kv_seq * head_dim * sizeof(T));

    // Process each batch and head
    for(int b = 0; b < batch; b++)
    {
        for(int h = 0; h < head_num; h++)
        {
            int offset_Q       = (b * head_num + h) * q_seq * head_dim;
            int offset_K       = (b * head_num + h) * kv_seq * head_dim;
            int offset_V       = (b * head_num + h) * kv_seq * head_dim;
            int offset_grad_O  = (b * head_num + h) * q_seq * head_dim;
            int offset_attn    = (b * head_num + h) * q_seq * kv_seq;
            int offset_mask    = mask ? (b * head_num + h) * q_seq * kv_seq : 0;
            int offset_dropout = dropout_mask ? (b * head_num + h) * q_seq * kv_seq : 0;

            const T* Q_bh          = Q + offset_Q;
            const T* K_bh          = K + offset_K;
            const T* V_bh          = V + offset_V;
            const T* grad_O_bh     = grad_O + offset_grad_O;
            const T* attn_bh       = attn_weights + offset_attn;
            const uint8_t* mask_bh = mask ? mask + offset_mask : nullptr;
            const T* dropout_bh    = dropout_mask ? dropout_mask + offset_dropout : nullptr;

            T* grad_Q_bh = grad_Q + offset_Q;
            T* grad_K_bh = grad_K + offset_K;
            T* grad_V_bh = grad_V + offset_V;

            // Step 1: grad_V = attn_weights^T @ grad_O
            // attn_weights: [q_seq, kv_seq], grad_O: [q_seq, head_dim] -> grad_V: [kv_seq,
            // head_dim]
            transpose(attn_bh, attn_T.data(), q_seq, kv_seq);
            matmul(attn_T.data(), grad_O_bh, grad_V_bh, kv_seq, q_seq, head_dim);

            // Step 2: grad_attn = grad_O @ V^T
            // grad_O: [q_seq, head_dim], V: [kv_seq, head_dim] -> grad_attn: [q_seq, kv_seq]
            transpose(V_bh, V_T.data(), kv_seq, head_dim);
            matmul(grad_O_bh, V_T.data(), grad_attn.data(), q_seq, head_dim, kv_seq);
            // Step 3: Dropout backward
            if(dropout_p > 0.0f && dropout_bh != nullptr)
            {
                for(int i = 0; i < q_seq * kv_seq; i++)
                {
                    grad_attn[i] = T(float(grad_attn[i]) * float(dropout_bh[i]) * dropout_scale);
                }
            }

            // Step 4: Softmax backward
            // grad_scores = (grad_attn * attn_weights) - attn_weights * sum(grad_attn *
            // attn_weights, dim=-1)
            for(int i = 0; i < q_seq * kv_seq; i++)
            {
                grad_scores[i] = T(float(grad_attn[i]) * float(attn_bh[i]));
            }

            sum_last_dim(grad_scores.data(), row_sums.data(), q_seq, kv_seq);

            for(int i = 0; i < q_seq; i++)
            {
                for(int j = 0; j < kv_seq; j++)
                {
                    int idx = i * kv_seq + j;
                    grad_scores[idx] =
                        T(float(grad_scores[idx]) - float(attn_bh[idx]) * float(row_sums[i]));
                }
            }

            // Step 5: Mask backward
            if(mask_bh != nullptr)
            {
                for(int i = 0; i < q_seq * kv_seq; i++)
                {
                    if(mask_bh[i] == 0)
                    {
                        grad_scores[i] = T(0.0f);
                    }
                }
            }

            // Step 6: grad_Q = grad_scores @ K / scale
            // grad_scores: [q_seq, kv_seq], K: [kv_seq, head_dim] -> grad_Q: [q_seq, head_dim]
            matmul(grad_scores.data(), K_bh, grad_Q_bh, q_seq, kv_seq, head_dim);
            for(int i = 0; i < q_seq * head_dim; i++)
            {
                grad_Q_bh[i] = T(float(grad_Q_bh[i]) * scale);
            }

            // Step 7: grad_K = grad_scores^T @ Q / scale
            // grad_scores: [q_seq, kv_seq], Q: [q_seq, head_dim] -> grad_K: [kv_seq, head_dim]
            transpose(grad_scores.data(), grad_scores_T.data(), q_seq, kv_seq);
            matmul(grad_scores_T.data(), Q_bh, grad_K_bh, kv_seq, q_seq, head_dim);
            for(int i = 0; i < kv_seq * head_dim; i++)
            {
                grad_K_bh[i] = T(float(grad_K_bh[i]) * scale);
            }
        }
    }
}

/**
 * Test run_attn_bwd_kernel correctness and bandwidth
 */
template <typename DataType, typename Config>
void test_run_attn_bwd_kernel(
    float dropout_p, int warmup_iters, int test_iters, bool check_correctness, bool dump_err)
{
    using Launcher = AttnBackwardKernelLauncher<DataType, Config>;

    constexpr int bs       = Config::bs;
    constexpr int head_num = Config::head_num;
    constexpr int seq_q    = Config::seq_q;
    constexpr int seq_kv   = Config::seq_kv;
    constexpr int head_dim = Config::head_dim;

    // Calculate sizes
    size_t size_Q            = bs * head_num * seq_q * head_dim;
    size_t size_K            = bs * head_num * seq_kv * head_dim;
    size_t size_V            = bs * head_num * seq_kv * head_dim;
    size_t size_grad_O       = bs * head_num * seq_q * head_dim;
    size_t size_attn_weights = bs * head_num * seq_q * seq_kv;
    size_t size_mask         = bs * head_num * seq_q * seq_kv;
    size_t size_dropout_mask = bs * head_num * seq_q * seq_kv;

    // Allocate host memory
    std::vector<DataType> h_Q(size_Q);
    std::vector<DataType> h_K(size_K);
    std::vector<DataType> h_V(size_V);
    std::vector<DataType> h_grad_O(size_grad_O);
    std::vector<DataType> h_attn_weights(size_attn_weights);
    std::vector<uint8_t> h_mask(size_mask);
    std::vector<DataType> h_dropout_mask(size_dropout_mask);
    std::vector<DataType> h_grad_Q_gpu(size_Q);
    std::vector<DataType> h_grad_K_gpu(size_K);
    std::vector<DataType> h_grad_V_gpu(size_V);
    std::vector<DataType> h_grad_Q_cpu(size_Q);
    std::vector<DataType> h_grad_K_cpu(size_K);
    std::vector<DataType> h_grad_V_cpu(size_V);

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
    for(size_t i = 0; i < size_grad_O; i++)
        h_grad_O[i] = DataType(dis(gen));

    // Initialize attention weights (normalized softmax output)
    for(int b = 0; b < bs; b++)
    {
        for(int h = 0; h < head_num; h++)
        {
            for(int i = 0; i < seq_q; i++)
            {
                float sum = 0.0f;
                for(int j = 0; j < seq_kv; j++)
                {
                    int idx             = ((b * head_num + h) * seq_q + i) * seq_kv + j;
                    h_attn_weights[idx] = DataType(std::abs(dis(gen)));
                    sum += float(h_attn_weights[idx]);
                }
                if(sum > 0.0f)
                {
                    for(int j = 0; j < seq_kv; j++)
                    {
                        int idx             = ((b * head_num + h) * seq_q + i) * seq_kv + j;
                        h_attn_weights[idx] = DataType(float(h_attn_weights[idx]) / sum);
                    }
                }
            }
        }
    }

    // Initialize mask (1 = keep, 0 = mask out)
    for(size_t i = 0; i < size_mask; i++)
    {
        h_mask[i] = Config::enable_mask ? (dis(gen) > 0.0f ? 1 : 0) : 1;
    }

    // Initialize dropout mask (1 = keep, 0 = drop)
    for(size_t i = 0; i < size_dropout_mask; i++)
    {
        h_dropout_mask[i] = Config::enable_dropout_mask
                                ? DataType(dis(gen) > dropout_p ? 1.0f : 0.0f)
                                : DataType(1.0f);
    }

    // Compute CPU reference
    float sqr_dk_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    attn_backward(h_Q.data(),
                  h_K.data(),
                  h_V.data(),
                  h_grad_O.data(),
                  h_attn_weights.data(),
                  Config::enable_mask ? h_mask.data() : nullptr,
                  Config::enable_dropout_mask ? h_dropout_mask.data() : nullptr,
                  dropout_p,
                  h_grad_Q_cpu.data(),
                  h_grad_K_cpu.data(),
                  h_grad_V_cpu.data(),
                  bs,
                  head_num,
                  seq_q,
                  seq_kv,
                  head_dim);

    // Allocate device memory
    DataType *d_Q, *d_K, *d_V, *d_grad_O, *d_attn_weights;
    uint8_t* d_mask;
    DataType* d_dropout_mask;
    DataType *d_grad_Q, *d_grad_K, *d_grad_V, *d_workspace;

    HIP_CHECK(hipMalloc(&d_Q, size_Q * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_K, size_K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_V, size_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_grad_O, size_grad_O * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_attn_weights, size_attn_weights * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_mask, size_mask * sizeof(uint8_t)));
    HIP_CHECK(hipMalloc(&d_dropout_mask, size_dropout_mask * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_grad_Q, size_Q * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_grad_K, size_K * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_grad_V, size_V * sizeof(DataType)));

    size_t workspace_size = Launcher::calc_workspace_size();
    HIP_CHECK(hipMalloc(&d_workspace, workspace_size));

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_Q, h_Q.data(), size_Q * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_K, h_K.data(), size_K * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), size_V * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_grad_O, h_grad_O.data(), size_grad_O * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_attn_weights,
                        h_attn_weights.data(),
                        size_attn_weights * sizeof(DataType),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_mask, h_mask.data(), size_mask * sizeof(uint8_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dropout_mask,
                        h_dropout_mask.data(),
                        size_dropout_mask * sizeof(DataType),
                        hipMemcpyHostToDevice));

    // Warmup runs
    for(int i = 0; i < warmup_iters; i++)
    {
        Launcher::run_attn_bwd_kernel(d_Q,
                                      d_K,
                                      d_V,
                                      d_grad_O,
                                      d_attn_weights,
                                      Config::enable_mask ? d_mask : nullptr,
                                      Config::enable_dropout_mask ? d_dropout_mask : nullptr,
                                      dropout_p,
                                      sqr_dk_scale,
                                      d_grad_Q,
                                      d_grad_K,
                                      d_grad_V,
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
        Launcher::run_attn_bwd_kernel(d_Q,
                                      d_K,
                                      d_V,
                                      d_grad_O,
                                      d_attn_weights,
                                      Config::enable_mask ? d_mask : nullptr,
                                      Config::enable_dropout_mask ? d_dropout_mask : nullptr,
                                      dropout_p,
                                      sqr_dk_scale,
                                      d_grad_Q,
                                      d_grad_K,
                                      d_grad_V,
                                      d_workspace);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    double avg_time_ms = elapsed_ms / test_iters;

    // Copy results back
    HIP_CHECK(
        hipMemcpy(h_grad_Q_gpu.data(), d_grad_Q, size_Q * sizeof(DataType), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(h_grad_K_gpu.data(), d_grad_K, size_K * sizeof(DataType), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(h_grad_V_gpu.data(), d_grad_V, size_V * sizeof(DataType), hipMemcpyDeviceToHost));

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
    size_t bytes_read =
        (size_Q + size_K + size_V + size_grad_O + size_attn_weights) * sizeof(DataType);
    if(Config::enable_mask)
        bytes_read += size_mask * sizeof(DataType);
    if(Config::enable_dropout_mask)
        bytes_read += size_dropout_mask * sizeof(DataType);

    size_t bytes_write    = (size_Q + size_K + size_V) * sizeof(DataType);
    size_t total_bytes    = bytes_read + bytes_write;
    double bandwidth_gbps = (total_bytes / 1e9) / (avg_time_ms / 1000.0);

    // Print results
    std::cout << "\n===== run_attn_bwd_kernel Test =====" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << bs << std::endl;
    std::cout << "  Heads: " << head_num << std::endl;
    std::cout << "  Q sequence length: " << seq_q << std::endl;
    std::cout << "  KV sequence length: " << seq_kv << std::endl;
    std::cout << "  Head dimension: " << head_dim << std::endl;
    std::cout << "  Dropout: " << (Config::enable_dropout_mask ? "enabled" : "disabled")
              << std::endl;
    std::cout << "  Mask: " << (Config::enable_mask ? "enabled" : "disabled") << std::endl;
    std::cout << std::endl;

    if(check_correctness)
    {
        std::cout << "Correctness:" << std::endl;
        check_results(h_grad_Q_gpu, h_grad_Q_cpu, "grad_Q");
        check_results(h_grad_K_gpu, h_grad_K_cpu, "grad_K");
        check_results(h_grad_V_gpu, h_grad_V_cpu, "grad_V");
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
    std::cout << "====================================\n" << std::endl;

    // Cleanup
    HIP_CHECK(hipFree(d_Q));
    HIP_CHECK(hipFree(d_K));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_grad_O));
    HIP_CHECK(hipFree(d_attn_weights));
    HIP_CHECK(hipFree(d_mask));
    HIP_CHECK(hipFree(d_dropout_mask));
    HIP_CHECK(hipFree(d_grad_Q));
    HIP_CHECK(hipFree(d_grad_K));
    HIP_CHECK(hipFree(d_grad_V));
    HIP_CHECK(hipFree(d_workspace));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

int main(int argc, char const* argv[])
{
    // Create test configuration with all parameters in one place
    using KernelConfig1 = FmhaKernelConfig<30720, 16, 2, 1, 256, 128, true, true>;
    using KernelConfig2 = FmhaKernelConfig<30720, 16, 1, 2, 256, 128, true, true>;
    using KernelConfig3 = FmhaKernelConfig<30720, 32, 2, 1, 128, 128, true, true>;
    using KernelConfig4 = FmhaKernelConfig<30720, 32, 1, 2, 128, 128, true, true>;

    // std::cout << "\n========== Testing with float ==========" << std::endl;
    test_run_attn_bwd_kernel<float, KernelConfig1>(0.3, 0, 1, true, false);
    test_run_attn_bwd_kernel<float, KernelConfig2>(0.3, 0, 1, true, false);
    test_run_attn_bwd_kernel<float, KernelConfig3>(0.3, 0, 1, true, false);
    test_run_attn_bwd_kernel<float, KernelConfig4>(0.3, 0, 1, true, false);

    std::cout << "\n========== Testing with bfloat16 ==========" << std::endl;
    test_run_attn_bwd_kernel<hip_bfloat16, KernelConfig1>(0.3, 0, 1, false, false);
    test_run_attn_bwd_kernel<hip_bfloat16, KernelConfig2>(0.3, 0, 1, false, false);
    test_run_attn_bwd_kernel<hip_bfloat16, KernelConfig3>(0.3, 0, 1, false, false);
    test_run_attn_bwd_kernel<hip_bfloat16, KernelConfig4>(0.3, 0, 1, false, false);

    return 0;
}