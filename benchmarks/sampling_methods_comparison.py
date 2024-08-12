# benchmarks/sampling_performance_comparison.py 
# This is an LLM (claude sonnet 3.5) generated code file.

import os
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import time

def method1_store_indexes(data, num_samples):
    """Sample without replacement using a set to store used indexes."""
    sampled_indexes = set()
    samples = []
    for _ in range(num_samples):
        while True:
            idx = np.random.randint(0, len(data))
            if idx not in sampled_indexes:
                sampled_indexes.add(idx)
                samples.append(data[idx])
                break
    return samples

def method2_check_previous(data, num_samples):
    """Sample without replacement by checking against previous samples."""
    samples = []
    for _ in range(num_samples):
        while True:
            sample = data[np.random.randint(0, len(data))]
            if sample.tolist() not in samples:
                samples.append(sample.tolist())
                break
    return samples

def profile_function(func, *args):
    """Profile the memory usage and execution time of a function."""
    start_time = time.time()
    mem_usage = memory_usage((func, args), interval=0.1, timeout=600)
    end_time = time.time()
    return max(mem_usage) - mem_usage[0], end_time - start_time

def run_scalability_comparison(data_size, max_samples, step):
    """Run a scalability comparison of the two sampling methods."""
    data = np.random.rand(data_size, 2)  # 2D data points
    sample_sizes = range(step, max_samples + step, step)

    method1_memory = []
    method1_time = []
    method2_memory = []
    method2_time = []

    for num_samples in sample_sizes:
        mem1, time1 = profile_function(method1_store_indexes, data, num_samples)
        mem2, time2 = profile_function(method2_check_previous, data, num_samples)

        method1_memory.append(mem1)
        method1_time.append(time1)
        method2_memory.append(mem2)
        method2_time.append(time2)

        print(f"Completed sampling {num_samples} out of {max_samples}")

    # Create the results directory if it doesn't exist
    results_dir = os.path.join('results', 'benchmarks')
    os.makedirs(results_dir, exist_ok=True)

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(sample_sizes, method1_memory, label='Method 1: Store Indexes')
    ax1.plot(sample_sizes, method2_memory, label='Method 2: Check Previous')
    ax1.set_xlabel('Number of Samples')
    ax1.set_ylabel('Peak Memory Usage (MiB)')
    ax1.set_title(f'Memory Usage Comparison (Data Size: {data_size})')
    ax1.legend()

    ax2.plot(sample_sizes, method1_time, label='Method 1: Store Indexes')
    ax2.plot(sample_sizes, method2_time, label='Method 2: Check Previous')
    ax2.set_xlabel('Number of Samples')
    ax2.set_ylabel('Execution Time (s)')
    ax2.set_title('Execution Time Comparison')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'scalability_comparison_{data_size}.png'))
    plt.close()

    # Print summary
    print(f"Data Size: {data_size}")
    print(f"Method 1 (Store Indexes) - Max Memory: {max(method1_memory):.2f} MiB, Max Time: {max(method1_time):.2f}s")
    print(f"Method 2 (Check Previous) - Max Memory: {max(method2_memory):.2f} MiB, Max Time: {max(method2_time):.2f}s")
    print(f"Results saved in {results_dir}")

if __name__ == "__main__":
    run_scalability_comparison(100000, 50000, 5000)