import pandas as pd
from data_colors import COLOR_MAP
from data_io import SyntheticDataset
import numpy as np


def process_and_save_data(image_path, output_path, normalize=False):
    """Processes the data from an image and saves it to a CSV file."""
    extractor = SyntheticDataset(image_path, COLOR_MAP)
    dataset: pd.DataFrame = extractor.extract_data()

    if normalize:
        print("Normalizing dataset...")
        normalized_coordinates: np.ndarray = extractor.normalize_coordinates(dataset)
        dataset["normalized_coordinates"] = [(float(x), float(y)) for x,y in normalized_coordinates]

    dataset.to_csv(output_path, index=False)
    extractor.plot_data(dataset, normalized=True)
    print(f"Processed dataset and saved to {output_path}")

# Example usage
if __name__ == "__main__":
    image_path = "data/dataset1.png"
    save_path = "data/dataset1.csv"
    process_and_save_data(image_path, save_path, normalize=True)

