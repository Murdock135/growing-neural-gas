import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from sklearn.preprocessing import LabelEncoder

# colormaps
colors = ['#ffd1df', 'salmon', 'red']
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
    result_matrix = np.hstack((result_matrix, np.zeros(shape=(col_n + padding_size, padding_size))))
    
    return result_matrix

class SyntheticDataset:
    def __init__(self, image_path):
        self.image_path = image_path
        self.dataset = None

    def get_dataset(self) -> pd.DataFrame:
        return self.dataset

    def get_image_path(self):
        return self.image_path

    def generate_data_from_img(self):
        # Open the image
        img = Image.open(self.image_path)

        # Convert the image to RGBA (if it's not already in that format)
        img = img.convert("RGBA")

        # Get the size of the image
        width, height = img.size

        # Prepare a list to hold coordinates and color of colored points
        data = []

        # Loop through each pixel in the image
        for i in range(width):
            for j in range(height):
                # Get the color of the pixel (excluding alpha channel for simplicity)
                r, g, b, a = img.getpixel((i, j))
                if (r, g, b) == (237, 28, 36):
                    data.append(((i, j), (r, g, b), "red"))
                if (r, g, b) == (255, 127, 29):
                    data.append(((i, j), (r, g, b), "orange"))
                if (r, g, b) == (34, 177, 76):
                    data.append(((i, j), (r, g, b), "green"))
                if (r, g, b) == (255, 242, 0) or (r, g, b) == (255, 201, 14):
                    data.append(((i, j), (r, g, b), "yellow"))
                if (r, g, b) == (63, 72, 204):
                    data.append(((i, j), (r, g, b), "blue"))

        dataset = pd.DataFrame(data, columns=["coordinate", "rgb", "class"])
        self.dataset = dataset

    def one_hot_enc(self, column):
        one_hot_df = pd.get_dummies(self.dataset, columns=[column])

        return one_hot_df

    def transform_to_numeric_classes(self, labels_col):
        encoder = LabelEncoder()
        df_numeric_classes = self.dataset.copy()
        df_numeric_classes[f"{labels_col}"] = encoder.fit_transform(
            df_numeric_classes[f"{labels_col}"]
        )

        return df_numeric_classes

    def plot_data(self):
        df = self.dataset

        plt.figure(figsize=(8, 6))

        for label in df["class"].unique():
            class_data = df[df["class"] == label]
            data_x = [coord[0] for coord in class_data["coordinate"]]
            data_y = [coord[1] for coord in class_data["coordinate"]]

            plt.scatter(data_x, data_y, label=f"Class {label}", alpha=0.8, s=15, marker='o')

        # Optional: Invert y-axis to match the image's original coordinate system
        # ax.invert_yaxis()

        # Optional: Add legend and titles
        plt.legend()
        plt.title("Dataset")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()
