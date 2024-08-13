"""
Adaptive Vector Quantization Algorithm Runner

This script runs various Adaptive Vector Quantization algorithms 
(Currently supported: NG, GNG) based on the provided configuration. It supports different data inputs and saves
the results in a timestamped directory.

Usage:
    python src/run.py config.toml --data data/sample_data.txt --output custom_results

Author: Qazi Zarif Ul Islam
Date: August 7, 2024
"""

from typing import Dict, Any
from utils import (
    load_config,
    load_data,
    run_quantizer,
    parse_arguments,
    create_save_path,
)

def main() -> None:
    # Parse command-line arguments
    args = parse_arguments()

    # Load and parse the configuration file
    config: Dict[str, Any] = load_config(args.config)

    # Load data
    if args.data:
        data = load_data(args.data)
    else:
        print("No data source provided. Using default data from 'data/dataset1.png'")
        data = load_data(config.get("data", "ufa_to_ufa/data/dataset1.png"))

    print(data)

    # Get the algorithm name (default to 'gng' if not specified)
    algorithm: str = config.get("algorithm", "gng")

    # Create a timestamp-based save path
    save_path: str = create_save_path(args.output, algorithm)

    print(f"Results will be saved in: {save_path}")

    # Run the quantizer
    run_quantizer(algorithm, data, save_path, config)

if __name__ == "__main__":
    main()