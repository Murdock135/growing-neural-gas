import numpy as np
import timeit

def create_matrix(size, zero_row_probability=0.1):
    """Create a 2D matrix with random 1's and 0's, and some all-zero rows."""
    matrix = np.random.choice([0, 1], size=(size, size), p=[0.5, 0.5])
    zero_rows = np.random.choice([True, False], size=size, p=[zero_row_probability, 1-zero_row_probability])
    matrix[zero_rows] = 0
    return matrix

def find_zero_rows_method1(matrix):
    return np.where(~matrix.any(axis=1))[0]

def find_zero_rows_method2(matrix):
    return np.nonzero(np.all(matrix == 0, axis=1))[0]

# Function to run benchmark
def run_benchmark(method, matrix, num_runs=1000):
    return timeit.timeit(lambda: method(matrix), number=num_runs)

# Matrix sizes to test
sizes = [100, 1000, 5000]

print("Performance Comparison:")
print("----------------------")

for size in sizes:
    print(f"\nMatrix size: {size}x{size}")
    
    matrix = create_matrix(size)
    
    time1 = run_benchmark(find_zero_rows_method1, matrix)
    time2 = run_benchmark(find_zero_rows_method2, matrix)
    
    print(f"Method 1 time: {time1:.6f} seconds")
    print(f"Method 2 time: {time2:.6f} seconds")
    print(f"Method 2 is {time1/time2:.2f}x {'faster' if time2 < time1 else 'slower'} than Method 1")

    # Verify both methods produce the same result
    result1 = find_zero_rows_method1(matrix)
    result2 = find_zero_rows_method2(matrix)
    print(f"Results match: {np.array_equal(result1, result2)}")

print("\nReadability and Usage Comparison:")
print("----------------------------------")
print("Method 1: np.where(~matrix.any(axis=1))[0]")
print("Method 2: np.nonzero(np.all(matrix == 0, axis=1))[0]")
print("\nMethod 1 explanation:")
print("- Uses the logical NOT (~) of .any(), which might be less intuitive")
print("- Relies on implicit boolean conversion of non-zero values")
print("\nMethod 2 explanation:")
print("- More explicit with the condition (== 0)")
print("- Uses np.all() which directly conveys 'all elements must meet this condition'")
print("- np.nonzero() is often considered more idiomatic for boolean arrays")