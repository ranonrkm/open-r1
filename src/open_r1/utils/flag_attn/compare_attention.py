import torch
import time
import numpy as np
import os
import sys
from tabulate import tabulate

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from open_r1.utils.flag_attn.benchmark_cplsh import benchmark_cplsh_attention
from open_r1.utils.flag_attn.benchmark_flash import benchmark_flash_attention

def compare_attention_methods(
    batch_size=1,
    num_heads=8,
    seq_len_q=1024,
    seq_len_kv=1024,
    head_dim=128,
    lsh_L=32,
    local_size=128,
    num_warmup=50,
    num_runs=100,
    device="cuda"
):
    """
    Compare CPLSH attention with standard flash attention.
    
    Args:
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len_q: Query sequence length
        seq_len_kv: Key/Value sequence length
        head_dim: Head dimension
        lsh_L: Number of LSH codes
        local_size: Local attention window size
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark iterations
        device: Device to run on
    """
    print(f"\n{'='*80}")
    print(f"Comparing attention methods with configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Query sequence length: {seq_len_q}")
    print(f"  Key/Value sequence length: {seq_len_kv}")
    print(f"  Head dimension: {head_dim}")
    print(f"  LSH codes: {lsh_L}")
    print(f"  Local size: {local_size}")
    print(f"{'='*80}\n")
    
    # Run CPLSH attention benchmark
    print("\nBenchmarking CPLSH attention...")
    cplsh_results = benchmark_cplsh_attention(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        head_dim=head_dim,
        lsh_L=lsh_L,
        local_size=local_size,
        num_warmup=num_warmup,
        num_runs=num_runs,
        device=device
    )
    
    # Run Flash attention benchmark
    print("\nBenchmarking Flash attention...")
    flash_results = benchmark_flash_attention(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        head_dim=head_dim,
        num_warmup=num_warmup,
        num_runs=num_runs,
        device=device
    )
    
    # Prepare comparison table
    headers = ["Metric", "CPLSH Attention", "Flash Attention", "Speedup"]
    table_data = [
        ["Forward (ms)", f"{cplsh_results['forward_mean']:.2f} ± {cplsh_results['forward_std']:.2f}", 
         f"{flash_results['forward_mean']:.2f} ± {flash_results['forward_std']:.2f}",
         f"{flash_results['forward_mean']/cplsh_results['forward_mean']:.2f}x"],
        ["Backward (ms)", f"{cplsh_results['backward_mean']:.2f} ± {cplsh_results['backward_std']:.2f}", 
         f"{flash_results['backward_mean']:.2f} ± {flash_results['backward_std']:.2f}",
         f"{flash_results['backward_mean']/cplsh_results['backward_mean']:.2f}x"],
        ["Total (ms)", f"{cplsh_results['forward_mean'] + cplsh_results['backward_mean']:.2f} ± {cplsh_results['forward_std'] + cplsh_results['backward_std']:.2f}", 
         f"{flash_results['forward_mean'] + flash_results['backward_mean']:.2f} ± {flash_results['forward_std'] + flash_results['backward_std']:.2f}",
         f"{(flash_results['forward_mean'] + flash_results['backward_mean'])/(cplsh_results['forward_mean'] + cplsh_results['backward_mean']):.2f}x"],
        ["TFLOPS", f"{cplsh_results['tflops']:.2f}", f"{flash_results['tflops']:.2f}", 
         f"{cplsh_results['tflops']/flash_results['tflops']:.2f}x"]
    ]
    
    print("\nComparison Results:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Calculate memory usage
    q_size = batch_size * num_heads * seq_len_q * head_dim * 2  # 2 bytes for float16
    k_size = batch_size * num_heads * seq_len_kv * head_dim * 2
    v_size = batch_size * num_heads * seq_len_kv * head_dim * 2
    q_codes_size = batch_size * num_heads * seq_len_q * lsh_L * 4  # 4 bytes for int32
    k_codes_size = batch_size * num_heads * seq_len_kv * lsh_L * 4
    
    total_memory_cplsh = (q_size + k_size + v_size + q_codes_size + k_codes_size) / (1024 * 1024)  # MB
    total_memory_flash = (q_size + k_size + v_size) / (1024 * 1024)  # MB
    
    print(f"\nMemory Usage:")
    print(f"  CPLSH Attention: {total_memory_cplsh:.2f} MB")
    print(f"  Flash Attention: {total_memory_flash:.2f} MB")
    print(f"  Additional memory for LSH codes: {total_memory_cplsh - total_memory_flash:.2f} MB")
    
    return {
        "cplsh_results": cplsh_results,
        "flash_results": flash_results,
        "memory_cplsh": total_memory_cplsh,
        "memory_flash": total_memory_flash
    }

if __name__ == "__main__":
    # Test different configurations
    configs = [
        {"seq_len_q": 4096, "seq_len_kv": 4096, "head_dim": 128},
        {"seq_len_q": 8192, "seq_len_kv": 8192, "head_dim": 128},
        {"seq_len_q": 16384, "seq_len_kv": 16384, "head_dim": 128},
        {"seq_len_q": 32768, "seq_len_kv": 32768, "head_dim": 128},
    ]
    
    for config in configs:
        compare_attention_methods(**config) 