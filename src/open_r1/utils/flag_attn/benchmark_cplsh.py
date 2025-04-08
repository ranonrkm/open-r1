import torch
import time
import numpy as np
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from open_r1.utils.flag_attn.cplsh_flag_attn import cplsh_attention

def benchmark_cplsh_attention(
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
    Benchmark CPLSH attention forward and backward passes.
    
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
    # Create random inputs
    q = torch.randn(batch_size, num_heads, seq_len_q, head_dim, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(batch_size, num_heads, seq_len_kv, head_dim, device=device, dtype=torch.float16, requires_grad=True)
    
    # Create random LSH codes (integers)
    q_codes = torch.randint(-128, 127, (batch_size, num_heads, seq_len_q, lsh_L), device=device, dtype=torch.int32)
    k_codes = torch.randint(-128, 127, (batch_size, num_heads, seq_len_kv, lsh_L), device=device, dtype=torch.int32)
    
    # Warmup
    print("Warming up...")
    for _ in range(num_warmup):
        out = cplsh_attention(q, k, v, q_codes, k_codes, local_size=local_size)
        loss = out.sum()
        loss.backward()
    
    # Forward pass benchmark
    print("\nBenchmarking forward pass...")
    forward_times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = cplsh_attention(q, k, v, q_codes, k_codes, local_size=local_size)
        torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - start)
    
    # Backward pass benchmark
    print("Benchmarking backward pass...")
    backward_times = []
    for _ in range(num_runs):
        out = cplsh_attention(q, k, v, q_codes, k_codes, local_size=local_size)
        loss = out.sum()
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_times.append(time.perf_counter() - start)
    
    # Compute statistics
    forward_mean = np.mean(forward_times) * 1000  # Convert to ms
    forward_std = np.std(forward_times) * 1000
    backward_mean = np.mean(backward_times) * 1000
    backward_std = np.std(backward_times) * 1000
    
    # Print results
    print("\nResults:")
    print(f"Forward pass: {forward_mean:.2f} ± {forward_std:.2f} ms")
    print(f"Backward pass: {backward_mean:.2f} ± {backward_std:.2f} ms")
    print(f"Total: {(forward_mean + backward_mean):.2f} ± {(forward_std + backward_std):.2f} ms")
    
    # Compute theoretical FLOPs
    flops = (
        batch_size * num_heads * (
            # Forward pass
            seq_len_q * seq_len_kv * head_dim +  # QK matmul
            seq_len_q * seq_len_kv +  # Softmax
            seq_len_q * seq_len_kv * head_dim +  # OV matmul
            # Backward pass
            2 * seq_len_q * seq_len_kv * head_dim +  # dQ and dK
            seq_len_q * seq_len_kv * head_dim  # dV
        )
    )
    
    # Compute achieved TFLOPS
    tflops = flops / ((forward_mean + backward_mean) / 1000) / 1e12
    print(f"\nTheoretical TFLOPS: {tflops:.2f}")
    
    return {
        "forward_mean": forward_mean,
        "forward_std": forward_std,
        "backward_mean": backward_mean,
        "backward_std": backward_std,
        "tflops": tflops
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
        print(f"\nTesting configuration: {config}")
        benchmark_cplsh_attention(**config) 