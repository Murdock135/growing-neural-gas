from matplotlib.lines import lineStyles
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import os

global_colors = ["#ffd1df", "salmon", "red"]
cmap = ListedColormap(global_colors)


def plot_NG(
    data,
    neurons,
    iter,
    epoch,
    current_sample=None,
    sample_counts: np.ndarray = None,
    connection_matrix: np.ndarray = None,
    color_dict: dict = None,
):
    colors: dict = (
        color_dict
        if color_dict
        else {
            "data": "0.8",
            "neurons": "red",
            "current_sample_facecolor": "green",
            "current_sample_edgecolor": "k",
            "connection": "k",
            "cmap_colors": global_colors,
        }
    )

    plt.clf()
    cmap_color_vals = sample_counts / max(sample_counts)
    cmap_colors = colors["cmap_colors"].copy()
    cmap_colors.insert(0, colors["data"])
    cmap = ListedColormap(cmap_colors)

    plt.scatter(
        data[:, 0], data[:, 1], c=cmap_color_vals, cmap=cmap, marker="o", label="Data"
    )
    plt.scatter(
        neurons[:, 0], neurons[:, 1], c=colors["neurons"], marker="o", label="Neurons"
    )
    if current_sample is not None:
        plt.scatter(
            current_sample[0],
            current_sample[1],
            facecolor=colors["current_sample_facecolor"],
            edgecolors=colors["current_sample_edgecolor"],
            marker="o",
            s=100,
            label="Current sample",
        )

    if connection_matrix is not None:
        n1s, n2s = np.nonzero(np.triu(connection_matrix, k=1))
        for n1, n2 in zip(n1s, n2s):
            neuron_1 = neurons[n1]
            neuron_2 = neurons[n2]
            x_coords = [neuron_1[0], neuron_2[0]]
            y_coords = [neuron_1[1], neuron_2[1]]
            plt.plot(x_coords, y_coords, color=colors["connection"], linestyle="-")

    plt.title(f"Epoch {epoch}\nIteration {iter}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Number of times sampled", cmap=cmap)
    plt.legend()
    plt.draw()


def save_fig(save_dir, epoch, iter):
    file_name = os.path.join(save_dir, str(epoch), f"iter_{iter}.png")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    plt.savefig(file_name)
