from growing_neural_gas import GrowingNeuralGas
from neural_gas import NeuralGas
from utils.data_io import load_data, create_save_path, normalize_data

def run_gng(config):
    data = load_data(config['data'])
    data = normalize_data(data)  # TODO:
    save_path = create_save_path(config['results_dir'], 'gng')
    
    model = GrowingNeuralGas(
        data=data,
        results_dir=save_path,
        lifetime=config.get('lifetime', "auto"),
        max_iter=config.get('max_iter', "auto"),
        epochs=config.get('epochs', 3),
        plot_interval=config.get('plot_interval', 100),
        sampling_without_replacement=config.get('sampling_without_replacement', True),
        neurons_n=config.get('neurons_n', 2),
        max_neurons=config.get('max_neurons', 1000),
        eps_b=config.get('eps_b', "auto"),
        eps_n=config.get('eps_n', "auto"),
        lambda_param=config.get('lambda_param', "auto"),
        alpha=config.get('alpha', "auto"),
        decay=config.get('decay', "auto"),
        **config.get('plotting_colors', {})
    )
    model.run()

def run_ng(config):
    data = load_data(config['data'])
    data = normalize_data(data)  # Example usage
    save_path = create_save_path(config['results_dir'], 'ng')
    
    model = NeuralGas(
        data=data,
        results_dir=save_path,
        lifetime=config.get('lifetime', "auto"),
        max_iter=config.get('max_iter', "auto"),
        epochs=config.get('epochs', 3),
        plot_interval=config.get('plot_interval', 100),
        sampling_without_replacement=config.get('sampling_without_replacement', True),
        neurons_n=config.get('neurons_n', 200),
        epsilon=config.get('epsilon', "auto"),
        lambda_param=config.get('lambda_param', "auto"),
        **config.get('plotting_colors', {})
    )
    model.run()
