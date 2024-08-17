# Neural Gas and Growing Neural Gas Implementation

## Usage

To run the algorithm run

```sh
python run.py --config path/to/config.toml
```

Note: Create a config.toml file with the appropriate parameters used in the algorithm you wish to use.

## Overview

This project contains Python implementations of the Neural Gas (NG) and Growing Neural Gas (GNG) algorithms, which are unsupervised learning methods for vector quantization. These algorithms are useful for finding a representative set of prototypes in a dataset, which can be used in various machine learning and data compression tasks.

## Project Structure

- **`run.py`**: The main entry point for executing the NG and GNG algorithms. It accepts a configuration file (in TOML format) to specify the parameters for the run.
  
- **`main.py`**: Contains the high-level functions for running the NG and GNG algorithms. It loads the data, initializes the appropriate model, and executes the training process.
  
- **`neural_gas.py`**: Implements the Neural Gas (NG) algorithm. The NG algorithm is a form of vector quantization that adapts the neurons' positions to better represent the data distribution.
  
- **`growing_neural_gas.py`**: Implements the Growing Neural Gas (GNG) algorithm. GNG is an extension of NG that dynamically adds and removes neurons based on the data, allowing it to better adapt to complex data structures.
  
- **`adaptive_vector_quantizer.py`**: A base class for vector quantization algorithms like NG and GNG. It contains common methods and properties used by both algorithms.

## Requirements

- Python 3.x
- numpy
- matplotlib
- tqdm
- toml

You can install the required packages using:

```sh
pip install -r requirements.txt
```

## Class Architecture

### 1. **AdaptiveVectorQuantizer (Abstract Base Class)**

- **Attributes:**
  - `data`: `np.ndarray` (Data for the algorithm)
  - `neurons_n`: `int` (Number of neurons)
  - `results_dir`: `str` (Directory to save results)
  - `plotting_colors`: `dict` (Colors for plotting)
  - `lifetime`: `int` (Lifetime of connections)
  - `max_iter`: `int` (Maximum iterations)
  - `epochs`: `int` (Number of epochs)
  - `plot_interval`: `int` (Interval for plotting)
  - `sampling_without_replacement`: `bool` (Sampling logic)
  - `sample_counts`: `np.ndarray` (Counts of samples)
  - `color_dict`: `dict` (Color dictionary for plotting)
  - `fig`, `ax`: Matplotlib figure and axes objects
  - `colorbar`: Matplotlib colorbar object
  - `neurons`: `np.ndarray` (Neurons array)
  - `connection_matrix`: `np.ndarray` (Matrix of connections between neurons)

- **Methods:**
  - `__init__(...)` (Constructor initializing the above attributes)
  - `run()`: Runs the vector quantization process
  - `update(i: int, x: np.ndarray)`: Abstract method for updating the model (to be implemented in subclasses)
  - `get_random_sequence(data: np.ndarray)`: Generates a random sequence for sampling
  - `create_neurons(neurons_n: int, dist: str)`: Initializes neurons based on a distribution
  - `increase_age(r_index: int, c_index: int)`: Increases the age of a connection
  - `remove_old_connections()`: Removes connections older than a specified lifetime
  - `set_plotting_colors(**colors_kwargs)`: Sets the colors for plotting
  - `plot_NG(...)`: Plots the neurons, connections, and data
  - `create_cbar(cmap, norm)`: Creates a colorbar for the plot

### 2. **NeuralGas (Derived from AdaptiveVectorQuantizer)**

- **Attributes:**
  - `epsilon`: `float` (Learning rate for updating neurons)
  - `lambda_param`: `float` (Time constant for neighborhood function)

- **Methods:**
  - `__init__(...)`: Initializes NeuralGas-specific attributes along with the parent class attributes
  - `update(i: int, x: np.ndarray)`: Implements the update logic specific to the Neural Gas algorithm

### 3. **GrowingNeuralGas (Derived from AdaptiveVectorQuantizer)**

- **Attributes:**
  - `max_neurons`: `int` (Maximum number of neurons)
  - `eps_b`: `float` (Learning rate for the nearest neuron)
  - `eps_n`: `float` (Learning rate for neighboring neurons)
  - `lambda_param`: `int` (Frequency of neuron insertion)
  - `alpha`: `float` (Error reduction factor)
  - `decay`: `float` (Decay factor for errors)
  - `errors`: `np.ndarray` (Array to store errors for each neuron)

- **Methods:**
  - `__init__(...)`: Initializes GNG-specific attributes along with the parent class attributes
  - `update(i: int, x: np.ndarray)`: Implements the update logic specific to the Growing Neural Gas algorithm
  - `delete_lonely_neurons()`: Removes neurons that have no connections
  - `insert_new_neuron()`: Inserts a new neuron into the network based on the error values

### 4. **Miscellaneous Modules and Utilities**

- **Modules:**
  - `utils.plot_utils`: Provides utilities for plotting (e.g., colors)
  - `utils.data_io`: Contains functions for loading data and creating save paths
  - `adaptive_vector_quantizer`: Base class for vector quantization algorithms
