"""
Deep Learning HPC Demo - Portfolio Project
This script demonstrates the structure and functionality of the HPC-ready deep learning framework.
"""

import sys
import os

def main():
    print("Deep Learning HPC Demo")
    print("======================")
    print()
    print("This project demonstrates a high-performance computing ready deep learning framework")
    print("with the following components:")
    print()
    print("1. Multiple model implementations (Flax/JAX and PyTorch/DeepSpeed)")
    print("2. Distributed training capabilities")
    print("3. Professional deployment setup with Ray Serve")
    print("4. Comprehensive testing suite")
    print("5. Configuration management")
    print()
    print("Directory Structure:")
    print("- src/models/: Model implementations")
    print("- src/training/: Training scripts")
    print("- src/deployment/: Deployment scripts")
    print("- config/: Configuration files")
    print("- tests/: Unit tests")
    print("- notebooks/: Jupyter notebooks for experimentation")
    print()
    print("To run this project:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Configure training in config/train_config.yaml")
    print("3. Run training: python src/training/train_hpc.py")
    print("4. Deploy model: python src/deployment/serve_ray.py")
    print()
    print("For detailed examples, see notebooks/demo_experiments.ipynb")

if __name__ == "__main__":
    main()