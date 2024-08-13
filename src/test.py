import pandas as pd
import numpy as np
import torch.nn as nn

foo = np.array([1,2,4])

print(foo)

bar = pd.DataFrame(foo)

class myNet():
    def __init__(self, input, output) -> None:
        fc1 = nn.Linear(input, output)


print("bar: ", bar)

import matplotlib.pyplot as plt
import numpy as np

# Example data
data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
sample_counts = np.array([10, 15, 5, 20, 25])  # Example count data

# Normalize sample_counts to range [0, 1] for colormap
norm = plt.Normalize(sample_counts.min(), sample_counts.max())

# Create a colormap
cmap = plt.cm.viridis  # You can choose other colormaps as well

# Plot the data points, using sample_counts to set the color
plt.scatter(data[:, 0], data[:, 1], c=sample_counts, cmap=cmap, norm=norm)

# Add a colorbar to show the mapping
plt.colorbar(label='Sample Count')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Points Colored by Sample Count')
plt.show()