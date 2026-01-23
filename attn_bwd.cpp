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

template <int BS, int HEAD_NUM, int SEQ_Q, int SEQ_KV, int HEAD_DIM>
struct fmha_kernel_config
{
    static constexpr int bs       = BS;
    static constexpr int head_num = HEAD_NUM;
    static constexpr int seq_q    = SEQ_Q;
    static constexpr int seq_kv   = SEQ_KV;
    static constexpr int head_dim = HEAD_DIM;
};

template <typename T, typename shape_config>
__global__ void kernel1(const T* attn_weights,
                        const T* grad_O,
                        const T* V,
                        T* grad_V,
                        T* workspace) // store grad_attn
{
    constexpr int seq_q    = shape_config::seq_q;
    constexpr int seq_kv   = shape_config::seq_kv;
    constexpr int head_dim = shape_config::head_dim;

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
        fetch_grad_O[i * head_dim + thread_id] = grad_O_ptr[i * head_dim + thread_id];
    }
    __syncthreads();
// compute grad_V = attn_weights^T @ grad_O
#pragma unroll
    for(int i = 0; i < seq_kv; i++)
    {
        T sum = T(0.0f);
#pragma unroll
        for(int j = 0; j < seq_q; j++)
        {
            sum += attn_weights_ptr[j * seq_kv + i] * fetch_grad_O[j * head_dim + thread_id];
        }
        grad_V_ptr[i * head_dim + thread_id] = sum;
    }
    __syncthreads();

    // compute grad_attn = grad_O @ V^T
    // Each thread computes partial sum over head_dim, then reduce
    __shared__ T reduce_buffer[head_dim];

#pragma unroll
    for(int i = 0; i < seq_q; i++)
    {
#pragma unroll
        for(int j = 0; j < seq_kv; j++)
        {
            // Each thread computes one element of the dot product
            T partial_sum = T(0.0f);
            partial_sum = fetch_grad_O[i * head_dim + thread_id] * V_ptr[j * head_dim + thread_id];
            reduce_buffer[thread_id] = partial_sum;
            __syncthreads();

            // Parallel reduction in shared memory
            for(int stride = head_dim / 2; stride > 0; stride >>= 1)
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

template <typename T, typename mask_type, typename shape_config, bool dropout_mask, bool mask>
__global__ void kernel2(const T* attn_weights,
                        const T* dropout_mask,
                        const mask_type* mask_matrix,
                        T* grad_attn,
                        T dropout_scale)
{
    const uint32_t block_id        = blockIdx.x;
    const uint32_t thread_id       = threadIdx.x;
    constexpr int seq_q            = shape_config::seq_q;
    constexpr int seq_kv           = shape_config::seq_kv;
    constexpr int bs_num_per_block = blockDim.x;
    static_assert(bs_num_per_block / (seq_kv * seq_kv) > 0 &&
                      bs_num_per_block % (seq_q * seq_kv) == 0,
                  "wrong!");
    const uint32_t cur_block_offset = block_id * bs_num_per_block + thread_id;
    __shared__ T tmp_grad_score[bs_num_per_block];
    constexpr int reduce_score_num = bs_num_per_block / seq_kv;
    __shared__ T reduce_grad_scorep[reduce_score_num];
    constexpr loop_num = bs_num_per_block / (seq_q * seq_kv);

    T grad_attn_value = grad_attn[cur_block_offset];
    if constexpr(dropout_mask)
    {
        grad_attn_value = grad_attn_value * dropout_mask[cur_block_offset] * dropout_scale;
    }
    T attn_weight = attn_weights[cur_block_offset];
    T grad_score  = grad_attn_value * attn_weight;
    if constexpr(mask)
    {
        mask_type cur_thread_mask = mask_matrix[cur_block_offset];
        if(cur_thread_mask)
            grad_score = T(0.0f);
    }
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
        reduce_grad_scorep[thread_id] = sum;
    }
    __syncthreads();
    grad_score -= attn_weight * reduce_grad_scorep[thread_id / seq_kv];
    grad_attn[cur_block_offset] = grad_score;
}

// Helper function: Matrix multiplication C = A @ B
// A: [rows_a, cols_a], B: [cols_a, cols_b], C: [rows_a, cols_b]
void matmul(const hip_bfloat16* A,
            const hip_bfloat16* B,
            hip_bfloat16* C,
            int rows_a,
            int cols_a,
            int cols_b)
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
            C[i * cols_b + j] = hip_bfloat16(sum);
        }
    }
}

// Helper function: Matrix transpose
void transpose(const hip_bfloat16* A, hip_bfloat16* A_T, int rows, int cols)
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
void sum_last_dim(const hip_bfloat16* A, hip_bfloat16* sums, int rows, int cols)
{
    for(int i = 0; i < rows; i++)
    {
        float sum = 0.0f;
        for(int j = 0; j < cols; j++)
        {
            sum += float(A[i * cols + j]);
        }
        sums[i] = hip_bfloat16(sum);
    }
}

/**
 * Multi-Head Attention Backward Pass (CPU Reference Implementation)
 *
 * @param Q: Query tensor [batch, head_num, q_seq, head_dim]
 * @param K: Key tensor [batch, head_num, kv_seq, head_dim]
 * @param V: Value tensor [batch, head_num, kv_seq, head_dim]
 * @param grad_O: Gradient of output [batch, head_num, q_seq, head_dim]
 * @param attn_weights: Attention weights from forward pass [batch, head_num, q_seq, kv_seq]
 * @param mask: Optional mask [batch, head_num, q_seq, kv_seq] (nullptr if not used)
 * @param dropout_mask: Optional dropout mask [batch, head_num, q_seq, kv_seq] (nullptr if not
 * used)
 * @param dropout_p: Dropout probability
 * @param grad_Q: Output gradient for Q [batch, head_num, q_seq, head_dim]
 * @param grad_K: Output gradient for K [batch, head_num, kv_seq, head_dim]
 * @param grad_V: Output gradient for V [batch, head_num, kv_seq, head_dim]
 * @param batch: Batch size
 * @param head_num: Number of heads
 * @param q_seq: Query sequence length
 * @param kv_seq: Key/Value sequence length
 * @param head_dim: Head dimension
 */
void attn_backward(const hip_bfloat16* Q,
                   const hip_bfloat16* K,
                   const hip_bfloat16* V,
                   const hip_bfloat16* grad_O,
                   const hip_bfloat16* attn_weights,
                   const hip_bfloat16* mask,
                   const hip_bfloat16* dropout_mask,
                   float dropout_p,
                   hip_bfloat16* grad_Q,
                   hip_bfloat16* grad_K,
                   hip_bfloat16* grad_V,
                   int batch,
                   int head_num,
                   int q_seq,
                   int kv_seq,
                   int head_dim)
{

    float scale         = 1.0f / std::sqrt(static_cast<float>(head_dim));
    float dropout_scale = (dropout_p > 0.0f) ? (1.0f / (1.0f - dropout_p)) : 1.0f;

    // Allocate temporary buffers
    std::vector<hip_bfloat16> V_T(kv_seq * head_dim);
    std::vector<hip_bfloat16> grad_attn(q_seq * kv_seq);
    std::vector<hip_bfloat16> grad_scores(q_seq * kv_seq);
    std::vector<hip_bfloat16> attn_T(kv_seq * q_seq);
    std::vector<hip_bfloat16> grad_scores_T(kv_seq * q_seq);
    std::vector<hip_bfloat16> row_sums(q_seq);
    std::vector<hip_bfloat16> K_T(head_dim * kv_seq);
    std::vector<hip_bfloat16> Q_T(head_dim * q_seq);

    // Initialize gradients to zero
    std::memset(grad_Q, 0, batch * head_num * q_seq * head_dim * sizeof(hip_bfloat16));
    std::memset(grad_K, 0, batch * head_num * kv_seq * head_dim * sizeof(hip_bfloat16));
    std::memset(grad_V, 0, batch * head_num * kv_seq * head_dim * sizeof(hip_bfloat16));

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

            const hip_bfloat16* Q_bh       = Q + offset_Q;
            const hip_bfloat16* K_bh       = K + offset_K;
            const hip_bfloat16* V_bh       = V + offset_V;
            const hip_bfloat16* grad_O_bh  = grad_O + offset_grad_O;
            const hip_bfloat16* attn_bh    = attn_weights + offset_attn;
            const hip_bfloat16* mask_bh    = mask ? mask + offset_mask : nullptr;
            const hip_bfloat16* dropout_bh = dropout_mask ? dropout_mask + offset_dropout : nullptr;

            hip_bfloat16* grad_Q_bh = grad_Q + offset_Q;
            hip_bfloat16* grad_K_bh = grad_K + offset_K;
            hip_bfloat16* grad_V_bh = grad_V + offset_V;

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
                    grad_attn[i] =
                        hip_bfloat16(float(grad_attn[i]) * float(dropout_bh[i]) * dropout_scale);
                }
            }

            // Step 4: Softmax backward
            // grad_scores = (grad_attn * attn_weights) - attn_weights * sum(grad_attn *
            // attn_weights, dim=-1)
            for(int i = 0; i < q_seq * kv_seq; i++)
            {
                grad_scores[i] = hip_bfloat16(float(grad_attn[i]) * float(attn_bh[i]));
            }

            sum_last_dim(grad_scores.data(), row_sums.data(), q_seq, kv_seq);

            for(int i = 0; i < q_seq; i++)
            {
                for(int j = 0; j < kv_seq; j++)
                {
                    int idx          = i * kv_seq + j;
                    grad_scores[idx] = hip_bfloat16(float(grad_scores[idx]) -
                                                    float(attn_bh[idx]) * float(row_sums[i]));
                }
            }

            // Step 5: Mask backward
            if(mask_bh != nullptr)
            {
                for(int i = 0; i < q_seq * kv_seq; i++)
                {
                    if(float(mask_bh[i]) == 0.0f)
                    {
                        grad_scores[i] = hip_bfloat16(0.0f);
                    }
                }
            }

            // Step 6: grad_Q = grad_scores @ K / scale
            // grad_scores: [q_seq, kv_seq], K: [kv_seq, head_dim] -> grad_Q: [q_seq, head_dim]
            matmul(grad_scores.data(), K_bh, grad_Q_bh, q_seq, kv_seq, head_dim);
            for(int i = 0; i < q_seq * head_dim; i++)
            {
                grad_Q_bh[i] = hip_bfloat16(float(grad_Q_bh[i]) * scale);
            }

            // Step 7: grad_K = grad_scores^T @ Q / scale
            // grad_scores: [q_seq, kv_seq], Q: [q_seq, head_dim] -> grad_K: [kv_seq, head_dim]
            transpose(grad_scores.data(), grad_scores_T.data(), q_seq, kv_seq);
            matmul(grad_scores_T.data(), Q_bh, grad_K_bh, kv_seq, q_seq, head_dim);
            for(int i = 0; i < kv_seq * head_dim; i++)
            {
                grad_K_bh[i] = hip_bfloat16(float(grad_K_bh[i]) * scale);
            }
        }
    }
}

/**
 * Test configuration structure
 */
struct AttnTestConfig
{
    int batch;
    int head_num;
    int q_seq;
    int kv_seq;
    int head_dim;
    float dropout_p;
    int warmup_iters;
    int test_iters;

    AttnTestConfig(int b      = 2,
                   int h      = 8,
                   int q      = 1,
                   int kv     = 2,
                   int d      = 64,
                   float dp   = 0.0f,
                   int warmup = 1,
                   int test   = 1)
        : batch(b),
          head_num(h),
          q_seq(q),
          kv_seq(kv),
          head_dim(d),
          dropout_p(dp),
          warmup_iters(warmup),
          test_iters(test)
    {
    }
};

/**
 * Run attn_backward computation with timing
 * Returns average execution time in milliseconds
 */
double run_attn_backward_computation(const AttnTestConfig& config)
{
    // Calculate sizes
    size_t size_Q      = config.batch * config.head_num * config.q_seq * config.head_dim;
    size_t size_K      = config.batch * config.head_num * config.kv_seq * config.head_dim;
    size_t size_V      = config.batch * config.head_num * config.kv_seq * config.head_dim;
    size_t size_grad_O = config.batch * config.head_num * config.q_seq * config.head_dim;
    size_t size_attn   = config.batch * config.head_num * config.q_seq * config.kv_seq;

    // Allocate memory
    std::vector<hip_bfloat16> Q(size_Q);
    std::vector<hip_bfloat16> K(size_K);
    std::vector<hip_bfloat16> V(size_V);
    std::vector<hip_bfloat16> grad_O(size_grad_O);
    std::vector<hip_bfloat16> attn_weights(size_attn);
    std::vector<hip_bfloat16> grad_Q(size_Q);
    std::vector<hip_bfloat16> grad_K(size_K);
    std::vector<hip_bfloat16> grad_V(size_V);

    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for(size_t i = 0; i < size_Q; i++)
        Q[i] = hip_bfloat16(dis(gen));
    for(size_t i = 0; i < size_K; i++)
        K[i] = hip_bfloat16(dis(gen));
    for(size_t i = 0; i < size_V; i++)
        V[i] = hip_bfloat16(dis(gen));
    for(size_t i = 0; i < size_grad_O; i++)
        grad_O[i] = hip_bfloat16(dis(gen));

    // Initialize attention weights (softmax output, values in [0, 1])
    for(size_t i = 0; i < size_attn; i++)
    {
        attn_weights[i] = hip_bfloat16(std::abs(dis(gen)));
    }
    // Normalize each row to sum to 1
    for(int b = 0; b < config.batch; b++)
    {
        for(int h = 0; h < config.head_num; h++)
        {
            for(int n = 0; n < config.q_seq; n++)
            {
                int offset =
                    (b * config.head_num + h) * config.q_seq * config.kv_seq + n * config.kv_seq;
                float sum = 0.0f;
                for(int m = 0; m < config.kv_seq; m++)
                {
                    sum += float(attn_weights[offset + m]);
                }
                if(sum > 0.0f)
                {
                    for(int m = 0; m < config.kv_seq; m++)
                    {
                        attn_weights[offset + m] =
                            hip_bfloat16(float(attn_weights[offset + m]) / sum);
                    }
                }
            }
        }
    }

    // Warmup runs
    for(int i = 0; i < config.warmup_iters; i++)
    {
        attn_backward(Q.data(),
                      K.data(),
                      V.data(),
                      grad_O.data(),
                      attn_weights.data(),
                      nullptr,
                      nullptr,
                      config.dropout_p,
                      grad_Q.data(),
                      grad_K.data(),
                      grad_V.data(),
                      config.batch,
                      config.head_num,
                      config.q_seq,
                      config.kv_seq,
                      config.head_dim);
    }

    // Timed runs
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < config.test_iters; i++)
    {
        attn_backward(Q.data(),
                      K.data(),
                      V.data(),
                      grad_O.data(),
                      attn_weights.data(),
                      nullptr,
                      nullptr,
                      config.dropout_p,
                      grad_Q.data(),
                      grad_K.data(),
                      grad_V.data(),
                      config.batch,
                      config.head_num,
                      config.q_seq,
                      config.kv_seq,
                      config.head_dim);
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate statistics
    std::chrono::duration<double, std::milli> elapsed = end - start;
    double avg_time_ms                                = elapsed.count() / config.test_iters;

    return avg_time_ms;
}

/**
 * Test and report bandwidth of attn_backward function
 */
void test_attn_backward_bandwidth(const AttnTestConfig& config)
{
    // Run computation and get timing
    double avg_time_ms = run_attn_backward_computation(config);

    // Calculate sizes for bandwidth computation
    size_t size_Q      = config.batch * config.head_num * config.q_seq * config.head_dim;
    size_t size_K      = config.batch * config.head_num * config.kv_seq * config.head_dim;
    size_t size_V      = config.batch * config.head_num * config.kv_seq * config.head_dim;
    size_t size_grad_O = config.batch * config.head_num * config.q_seq * config.head_dim;
    size_t size_attn   = config.batch * config.head_num * config.q_seq * config.kv_seq;

    // Calculate memory traffic (in bytes)
    // Reads: Q, K, V, grad_O, attn_weights
    // Writes: grad_Q, grad_K, grad_V
    size_t bytes_read  = (size_Q + size_K + size_V + size_grad_O + size_attn) * sizeof(float);
    size_t bytes_write = (size_Q + size_K + size_V) * sizeof(float);
    size_t total_bytes = bytes_read + bytes_write;

    double bandwidth_gbps = (total_bytes / 1e9) / (avg_time_ms / 1000.0);

    // Print results
    std::cout << "===== Attention Backward Bandwidth Test =====" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << config.batch << std::endl;
    std::cout << "  Heads: " << config.head_num << std::endl;
    std::cout << "  Q sequence length: " << config.q_seq << std::endl;
    std::cout << "  KV sequence length: " << config.kv_seq << std::endl;
    std::cout << "  Head dimension: " << config.head_dim << std::endl;
    std::cout << std::endl;
    std::cout << "Memory:" << std::endl;
    std::cout << "  Total data read: " << std::fixed << std::setprecision(2) << bytes_read / 1e6
              << " MB" << std::endl;
    std::cout << "  Total data write: " << bytes_write / 1e6 << " MB" << std::endl;
    std::cout << "  Total data transfer: " << total_bytes / 1e6 << " MB" << std::endl;
    std::cout << std::endl;
    std::cout << "Performance:" << std::endl;
    std::cout << "  Average time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms"
              << std::endl;
    std::cout << "  Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth_gbps << " GB/s"
              << std::endl;
    std::cout << "=============================================" << std::endl;
}

/**
 * Test kernel1 bandwidth
 */
template <typename Config>
void test_kernel1_bandwidth(const AttnTestConfig& config)
{
    using DataType = hip_bfloat16;

    // Validate config matches template parameters
    if(config.q_seq != Config::seq_q || config.kv_seq != Config::seq_kv ||
       config.head_dim != Config::head_dim)
    {
        std::cerr << "Error: Config mismatch with template parameters!" << std::endl;
        return;
    }

    constexpr int seq_q    = Config::seq_q;
    constexpr int seq_kv   = Config::seq_kv;
    constexpr int head_dim = Config::head_dim;

    const int merge_bs = config.batch * config.head_num;

    // Calculate sizes
    size_t size_attn_weights = merge_bs * seq_q * seq_kv;
    size_t size_grad_O       = merge_bs * seq_q * head_dim;
    size_t size_V            = merge_bs * seq_kv * head_dim;
    size_t size_grad_V       = merge_bs * seq_kv * head_dim;
    size_t size_workspace    = merge_bs * seq_q * seq_kv;

    // Allocate host memory
    std::vector<DataType> h_attn_weights(size_attn_weights);
    std::vector<DataType> h_grad_O(size_grad_O);
    std::vector<DataType> h_V(size_V);
    std::vector<DataType> h_grad_V(size_grad_V);
    std::vector<DataType> h_workspace(size_workspace);

    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for(size_t i = 0; i < size_attn_weights; i++)
        h_attn_weights[i] = hip_bfloat16(std::abs(dis(gen)));
    for(size_t i = 0; i < size_grad_O; i++)
        h_grad_O[i] = hip_bfloat16(dis(gen));
    for(size_t i = 0; i < size_V; i++)
        h_V[i] = hip_bfloat16(dis(gen));

    // Normalize attention weights (each row sums to 1)
    for(int mb = 0; mb < merge_bs; mb++)
    {
        for(int i = 0; i < seq_q; i++)
        {
            float sum = 0.0f;
            for(int j = 0; j < seq_kv; j++)
            {
                sum += float(h_attn_weights[mb * seq_q * seq_kv + i * seq_kv + j]);
            }
            if(sum > 0.0f)
            {
                for(int j = 0; j < seq_kv; j++)
                {
                    h_attn_weights[mb * seq_q * seq_kv + i * seq_kv + j] = hip_bfloat16(
                        float(h_attn_weights[mb * seq_q * seq_kv + i * seq_kv + j]) / sum);
                }
            }
        }
    }

    // Allocate device memory
    DataType *d_attn_weights, *d_grad_O, *d_V, *d_grad_V, *d_workspace;
    HIP_CHECK(hipMalloc(&d_attn_weights, size_attn_weights * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_grad_O, size_grad_O * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_V, size_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_grad_V, size_grad_V * sizeof(DataType)));
    HIP_CHECK(hipMalloc(&d_workspace, size_workspace * sizeof(DataType)));

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_attn_weights,
                        h_attn_weights.data(),
                        size_attn_weights * sizeof(DataType),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(
        d_grad_O, h_grad_O.data(), size_grad_O * sizeof(DataType), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_V, h_V.data(), size_V * sizeof(DataType), hipMemcpyHostToDevice));

    // Kernel launch configuration
    dim3 grid(merge_bs);  // Each block handles one merged batch
    dim3 block(head_dim); // Each block has head_dim threads

    // Warmup runs
    for(int i = 0; i < config.warmup_iters; i++)
    {
        kernel1<DataType, Config>
            <<<grid, block>>>(d_attn_weights, d_grad_O, d_V, d_grad_V, d_workspace);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Timed runs
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    for(int i = 0; i < config.test_iters; i++)
    {
        kernel1<DataType, Config>
            <<<grid, block>>>(d_attn_weights, d_grad_O, d_V, d_grad_V, d_workspace);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    double avg_time_ms = elapsed_ms / config.test_iters;

    // Copy results back (optional, for verification)
    HIP_CHECK(hipMemcpy(
        h_grad_V.data(), d_grad_V, size_grad_V * sizeof(DataType), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(
        h_workspace.data(), d_workspace, size_workspace * sizeof(DataType), hipMemcpyDeviceToHost));

    // Calculate bandwidth
    // Reads: attn_weights, grad_O, V (multiple times in loops)
    // Writes: grad_V, workspace
    size_t bytes_read  = (size_attn_weights + size_grad_O + size_V) * sizeof(DataType);
    size_t bytes_write = (size_grad_V + size_workspace) * sizeof(DataType);
    size_t total_bytes = bytes_read + bytes_write;

    double bandwidth_gbps = (total_bytes / 1e9) / (avg_time_ms / 1000.0);

    // Print results
    std::cout << "===== Kernel1 Bandwidth Test =====" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << config.batch << std::endl;
    std::cout << "  Heads: " << config.head_num << std::endl;
    std::cout << "  Merged BS: " << merge_bs << std::endl;
    std::cout << "  Q sequence length: " << seq_q << std::endl;
    std::cout << "  KV sequence length: " << seq_kv << std::endl;
    std::cout << "  Head dimension: " << head_dim << std::endl;
    std::cout << std::endl;
    std::cout << "Kernel Launch:" << std::endl;
    std::cout << "  Grid: (" << merge_bs << ")" << std::endl;
    std::cout << "  Block: (" << head_dim << ")" << std::endl;
    std::cout << std::endl;
    std::cout << "Memory:" << std::endl;
    std::cout << "  Total data read: " << std::fixed << std::setprecision(2) << bytes_read / 1e6
              << " MB" << std::endl;
    std::cout << "  Total data write: " << bytes_write / 1e6 << " MB" << std::endl;
    std::cout << "  Total data transfer: " << total_bytes / 1e6 << " MB" << std::endl;
    std::cout << std::endl;
    std::cout << "Performance:" << std::endl;
    std::cout << "  Average time: " << std::fixed << std::setprecision(3) << avg_time_ms << " ms"
              << std::endl;
    std::cout << "  Bandwidth: " << std::fixed << std::setprecision(2) << bandwidth_gbps << " GB/s"
              << std::endl;
    std::cout << "====================================" << std::endl;

    // Cleanup
    HIP_CHECK(hipFree(d_attn_weights));
    HIP_CHECK(hipFree(d_grad_O));
    HIP_CHECK(hipFree(d_V));
    HIP_CHECK(hipFree(d_grad_V));
    HIP_CHECK(hipFree(d_workspace));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
}

int main(int argc, char const* argv[])
{
    // Create test configuration
    AttnTestConfig config(30720, 16, 1, 2, 128, 0.0f, 0, 1);

    // Test CPU reference implementation
    // test_attn_backward_bandwidth(config);

    // std::cout << "\n\n";

    // Test GPU kernel1
    // Define config type matching the test parameters
    using KernelConfig = fmha_kernel_config<30720, 16, 1, 2, 128>;
    test_kernel1_bandwidth<KernelConfig>(config);

    return 0;
}