import numpy as np
import time

def method1_list_comprehension(list_of_lists):
    return [np.array(sublist) for sublist in list_of_lists]

def method2_map_function(list_of_lists):
    return list(map(np.array, list_of_lists))

def method3_frompyfunc(list_of_lists):
    return np.frompyfunc(np.array, 1, 1)(list_of_lists).tolist()

def method4_direct_numpy(list_of_lists):
    return np.array(list_of_lists)

def benchmark(func, list_of_lists, num_runs=100):
    start_time = time.time()
    for _ in range(num_runs):
        result = func(list_of_lists)
    end_time = time.time()
    return (end_time - start_time) / num_runs

# Generate test data
n_lists = 10000
list_length = 100
list_of_lists = [[np.random.randint(0, 100) for _ in range(list_length)] for _ in range(n_lists)]

# Run benchmarks
methods = [
    ("List Comprehension", method1_list_comprehension),
    ("Map Function", method2_map_function),
    ("NumPy frompyfunc", method3_frompyfunc),
    ("Direct NumPy", method4_direct_numpy)
]

for name, method in methods:
    avg_time = benchmark(method, list_of_lists)
    print(f"{name}: {avg_time:.6f} seconds")

# Verify results are the same
results = [method(list_of_lists) for _, method in methods]
# FIXME:
print("\nAll results identical:", all(np.array_equal(results[0][i], result[i]) for result in results[1:] for i in range(len(result))))