import numpy as np
from matplotlib.colors import ListedColormap
import toml
from data_utils import SyntheticDataset
from typing import Dict, Any
import argparse
from datetime import datetime
import os

# colormaps
colors = ["#ffd1df", "salmon", "red"]
cmap = ListedColormap(colors)

def zero(a):
    """Finds the idx of zeros in an array. This function is a complement of np.nonzero()"""
    mask = a == 0
    zero_idxs: tuple = np.nonzero(mask)
    return zero_idxs


def search_datapoint(arr: list, element: list):
    assert isinstance(arr, list), "Array to search datapoint in should be a list"
    assert isinstance(element, list), "Datapoint should be of class: list"

    if element in arr:
        return True
    else:
        return False


def add_padding(matrix: np.ndarray, padding_size: int = 1) -> np.ndarray:
    """
    Adds padding to a square matrix by adding rows and columns filled with zeros.

    Parameters:
    matrix (np.ndarray): The original square matrix.
    padding_size (int): The number of rows and columns to add. Default is 1.

    Returns:
    np.ndarray: The resulting matrix after adding the padding.
    """

    # Check if the input matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The input matrix must be square.")

    # Number of columns (elements in a row) in the original matrix
    row_n = matrix.shape[1]
    # Number of rows (elements in a column) in the original matrix
    col_n = matrix.shape[0]

    # Add new rows
    result_matrix = np.vstack((matrix, np.zeros(shape=(padding_size, row_n))))

    # Add new columns
    result_matrix = np.hstack(
        (result_matrix, np.zeros(shape=(col_n + padding_size, padding_size)))
    )

    return result_matrix


def load_config(config_path: str) -> Dict:
    """Load and Parse TOML file"""
    with open(config_path, "r") as f:
        config = toml.load(f)
    return config


def load_data(data_source: str) -> np.ndarray:
    """
    Load data from a file, generate random data, or use image-based data generation.

    Args:
        data_source (str): Path to data file, image file, or 'random' for random data.

    Returns:
        np.ndarray: 2D array of data points.
    """
    if data_source.lower() == "random":
        # Generate random data
        np.random.seed(0)
        return np.random.rand(1000, 2)
    elif data_source.lower().endswith((".png", ".jpg", ".jpeg")):
        # Use SyntheticDataset for image-based data generation
        synthetic_data = SyntheticDataset(data_source)
        synthetic_data.generate_data_from_img()
        df = synthetic_data.get_dataset()

        # Extract coordinates from the dataset
        coordinates = np.array(df["coordinate"].tolist())

        # Normalize the coordinates to [0, 1] range
        coordinates = (coordinates - coordinates.min(axis=0)) / (
            coordinates.max(axis=0) - coordinates.min(axis=0)
        )

        return coordinates
    else:
        # Load data from file
        return np.loadtxt(data_source)


def run_quantizer(
    algorithm: str, data: np.ndarray, output_path: str, config: Dict[str, Any]
) -> None:
    """
    Run the specified quantization algorithm.

    Args:
        algorithm (str): 'ng' for Neural Gas, 'gng' for Growing Neural Gas.
        data (np.ndarray): Input data.
        output_path (str): Path to save output files.
        config (Dict[str, Any]): Configuration dictionary.
    """
    from neural_gas import NeuralGas
    from growing_neural_gas import GrowingNeuralGas

    # Common parameters for both algorithms
    common_params = {
        "data": data,
        "fig_save_path": output_path,
        "max_iterations": config.get('max_iterations', 'auto'),
        "epochs": config.get('epochs', 3),
        "plot_interval": config.get('plot_interval', 100),
    }
    # FIXME:
    if algorithm == "ng":
        # Create and run Neural Gas
        quantizer = NeuralGas(
            **common_params,
            neurons_n=config.get('neurons_n', 200),
            epsilon=config["epsilon"],
            lambda_param=config["lambda"],
            lifetime=config["lifetime"],
        )
    elif algorithm == "gng":
        # Create and run Growing Neural Gas
        quantizer = GrowingNeuralGas(
            **common_params,
            neurons_n=config.get('neurons_n', 2),
            eps_b=config["eps_b"],
            eps_n=config["eps_n"],
            lifetime=config["lifetime"],
            alpha=config["alpha"],
            decay=config["decay"],
            lambda_param=config["lambda"],
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    quantizer.run()


def parse_arguments() -> argparse.Namespace:
    """
    Set up and parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run Adaptive Vector Quantization algorithms."
    )
    parser.add_argument("config", help="Path to TOML configuration file")
    parser.add_argument(
        "--data",
        default="c:/Users/Zayan/Documents/code/personal_repos/ufa_to_ufa/data/dataset1.png",
        help="Path to data file, image file, or 'random' for random data generation.",
    )
    parser.add_argument(
        "--output", default="results", help="Base path for output directory"
    )

    return parser.parse_args()


def create_save_path(base_path: str, algorithm: str) -> str:
    """
    Create a timestamped save path for the results.

    Args:
        base_path (str): Base directory for saving results.
        algorithm (str): Name of the algorithm being used.

    Returns:
        str: Full path to the save directory.
    """
    timestamp = datetime.now().strftime("%m-%d-%Y--%H-%M-%S")
    save_path = os.path.join(base_path, f"{algorithm.upper()}_{timestamp}")

    # Ensure the save directory exists
    os.makedirs(save_path, exist_ok=True)

    return save_path
