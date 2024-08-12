import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder

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
