import matplotlib.pyplot as plt
import imageio
import os
from datetime import datetime

sample_count_colors = ["#ffd1df", "salmon", "red"]
NG_colors = {
                "data": "0.8",
                "neurons": "k",
                "current_sample_facecolor": "green",
                "current_sample_edgecolor": "k",
                "connection": "k",
                "sample_count_colors": sample_count_colors,
            }

def save_fig(save_dir, epoch, iter):
    file_name = os.path.join(save_dir, str(epoch), f"iter_{iter}.png")
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    plt.savefig(file_name)


def _create_gif(figs_path, current_epoch) -> None:
    """Create a GIF from the saved plots for the current epoch."""
    figures = []
    results_for_epoch = os.path.join(figs_path, str(current_epoch))
    filenames = sorted(os.listdir(results_for_epoch))

    for f in filenames:
        filepath = os.path.join(figs_path, str(current_epoch), f)
        figures.append(imageio.imread(filepath))

    gif_path = os.path.join(
        results_for_epoch,
        f"epoch_{current_epoch}.gif",
    )
    imageio.mimsave(gif_path, figures, duration=0.1)
