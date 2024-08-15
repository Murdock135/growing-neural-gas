"""Author: Qazi Zarif Ul Islam

Entry point for running the Growing Neural Gas (GNG) or Neural Gas (NG) algorithm based on a TOML configuration file.

Usage:
    python run.py --config path/to/config.toml

Arguments:
    --config (str): Path to the TOML configuration file."""
import argparse
import toml
from main import run_gng, run_ng

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Growing Neural Gas or Neural Gas Algorithm")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (JSON format)")
    return parser.parse_args()

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as config_file:
        config = toml.load(config_file)
    return config

def main():
    args = parse_arguments()
    config = load_config(args.config)

    if config['algorithm'] == 'gng':
        run_gng(config)
    elif config['algorithm'] == 'ng':
        run_ng(config)

if __name__ == "__main__":
    main()
