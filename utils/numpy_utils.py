import numpy as np
from typing import Dict, Any


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




