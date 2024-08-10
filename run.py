from src.hebbian_algorithms import AdaptiveVectorQuantizer
from src.utils import SyntheticDataset
import pandas as pd
import toml
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

if __name__ == "__main__":
    print("Start")
    # Load configuration from TOML file
    config = toml.load("config.toml")

    data_image = (
        "C:/Users/Zayan/Documents/code/personal_repos/ufa_to_ufa/assets/dataset1.png"
    )
    
    print("Creating dataset")
    # generate_data_from_img
    dataset = SyntheticDataset(data_image)
    dataset.generate_data_from_img()
    df = dataset.get_dataset()
    data = np.array(df["coordinate"].to_list())
    print(data[:5])

    # display the data
    # dataset.plot_data()

    # hyperparameters
    neurons_n = config["neurons_n"]
    max_iter = config["max_iter"]
    lambda_i = 30
    lambda_f = 0.01
    eps_i = 0.3
    eps_f = 0.05
    T_i = 2
    T_f = 5


    timestamp = datetime.now().strftime("%m-%d-%Y--%H-%M-%S")
    save_path = f"results/NeuralGas_{timestamp}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # run NeuralGas
    algorithm_instance = AdaptiveVectorQuantizer(
        data=data,
        neurons_n=neurons_n,
        lifetime=T_i,
        epsilon=eps_f,
        _lambda=lambda_f,
        fig_save_path=save_path,
    )

    algorithm_instance.run()
