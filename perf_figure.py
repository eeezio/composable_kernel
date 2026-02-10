import matplotlib.pyplot as plt
import numpy as np

# 数据
seq_kv = list(range(2, 17))

# 您的kernel数据 (从输出中提取的平均时间)
hip_kernel_times = [
    0.422, 0.587, 0.768, 0.995, 1.209, 1.397, 1.531, 1.663,
    1.820, 1.987, 2.109, 2.261, 2.421, 2.591, 2.759
]

# JAX kernel数据 (mean_steptime_ms)
jax_kernel_times = [
    0.504, 0.659, 0.916, 1.029, 1.211, 1.418, 1.597, 1.778,
    1.953, 2.097, 2.333, 2.496, 4.392, 4.959, 4.646
]

# 计算加速比
speedup = [jax / hips for jax, hips in zip(jax_kernel_times, hip_kernel_times)]

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 子图1: 绝对性能对比
ax1.plot(seq_kv, hip_kernel_times, marker='o', linewidth=2, markersize=8, 
         label='HIP Kernel', color='#2E86AB')
ax1.plot(seq_kv, jax_kernel_times, marker='s', linewidth=2, markersize=8, 
         label='JAX Kernel', color='#A23B72')
ax1.set_xlabel('Sequence Length (KV)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Execution Time (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Attention Forward Kernel Performance Comparison\n(Batch=30720, Heads=32, HeadDim=128, SeqQ=1)', 
              fontsize=13, fontweight='bold', pad=15)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xticks(seq_kv)

# 添加数值标签
for i, (x, y1, y2) in enumerate(zip(seq_kv, hip_kernel_times, jax_kernel_times)):
    if i % 2 == 0:  # 每隔一个标注，避免拥挤
        ax1.annotate(f'{y1:.2f}', (x, y1), textcoords="offset points", 
                    xytext=(0,8), ha='center', fontsize=8, color='#2E86AB')
        ax1.annotate(f'{y2:.2f}', (x, y2), textcoords="offset points", 
                    xytext=(0,-15), ha='center', fontsize=8, color='#A23B72')

# 子图2: 加速比
ax2.plot(seq_kv, speedup, marker='D', linewidth=2.5, markersize=8, 
         color='#06A77D', label='Speedup (JAX/HIP)')
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline (1.0x)')
ax2.set_xlabel('Sequence Length (KV)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
ax2.set_title('Speedup: HIP Kernel vs JAX\n(Higher is Better)', 
              fontsize=13, fontweight='bold', pad=15)
ax2.legend(fontsize=11, loc='upper left')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xticks(seq_kv)
ax2.set_ylim([0.5, max(speedup) * 1.1])

# 添加加速比数值标签
for i, (x, y) in enumerate(zip(seq_kv, speedup)):
    if i % 2 == 0:
        ax2.annotate(f'{y:.2f}x', (x, y), textcoords="offset points", 
                    xytext=(0,8), ha='center', fontsize=9, 
                    color='#06A77D', fontweight='bold')

# 填充优势区域
ax2.fill_between(seq_kv, 1.0, speedup, where=[s > 1.0 for s in speedup], 
                 alpha=0.2, color='green', label='HIP Kernel Faster')
ax2.fill_between(seq_kv, speedup, 1.0, where=[s < 1.0 for s in speedup], 
                 alpha=0.2, color='red', label='JAX Faster')

plt.tight_layout()
plt.savefig('/root/composable_kernel/performance_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Performance comparison plot saved to: performance_comparison.png")

# 打印统计信息
print("\n" + "="*60)
print("PERFORMANCE SUMMARY")
print("="*60)
print(f"{'SeqKV':<8} {'HIP (ms)':<12} {'JAX (ms)':<12} {'Speedup':<10} {'Status'}")
print("-"*60)
for seq, hips, jax, sp in zip(seq_kv, hip_kernel_times, jax_kernel_times, speedup):
    status = "✓ Faster" if sp > 1.0 else "✗ Slower"
    print(f"{seq:<8} {hips:<12.3f} {jax:<12.3f} {sp:<10.2f}x {status}")
print("-"*60)
avg_speedup = np.mean(speedup)
print(f"Average Speedup: {avg_speedup:.2f}x")
print(f"Max Speedup: {max(speedup):.2f}x (at SeqKV={seq_kv[speedup.index(max(speedup))]})")
print(f"Min Speedup: {min(speedup):.2f}x (at SeqKV={seq_kv[speedup.index(min(speedup))]})")
win_count = sum(1 for s in speedup if s > 1.0)
print(f"HIP kernel wins: {win_count}/{len(speedup)} cases ({win_count/len(speedup)*100:.1f}%)")
print("="*60)
