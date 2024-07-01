from src.ng_alg import NeuralGas
from src.utils import SyntheticDataset
import pandas as pd
import toml

# Load configuration from TOML file
config = toml.load("config.toml")

data_image = (
    "C:/Users/Zayan/Documents/code/personal_repos/ufa_to_ufa/assets/dataset1.png"
)
print(type(data_image))

dataset = SyntheticDataset(data_image)
dataset.generate_data_from_img()
df_numeric_classes = dataset.transform_to_numeric_classes("class")
df = dataset.get_dataset()

print(df.head())
