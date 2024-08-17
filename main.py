from growing_neural_gas import GrowingNeuralGas
from neural_gas import NeuralGas
from utils.data_io import load_data, create_save_path


def run_gng(config):
    data = load_data(config["data"], normalize=True)
    save_path = create_save_path(config["results_dir"], "gng")

    model = GrowingNeuralGas(
        data=data, results_dir=save_path, lifetime=config.get("lifetime", 10)
    )
    model.run()


def run_ng(config):
    data = load_data(config["data"], normalize=True)
    save_path = create_save_path(config["results_dir"], "ng")

    model = NeuralGas(
        data=data,
        results_dir=save_path,
        lifetime=config.get("lifetime", 10),
    )
    model.run()
