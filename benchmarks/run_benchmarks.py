# Benchmarking script for Deep Learning HPC DEMO

import time
import jax
import jax.numpy as jnp
import torch
import numpy as np
import pandas as pd
import argparse
import json
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Import our models
from src.models.flax_mlp import FlaxMLP, FlaxMLPConfig
from src.models.flax_cnn import FlaxCNN, FlaxCNNConfig
from src.models.torch_deepspeed_mlp import TorchDeepSpeedMLP, TorchMLPConfig
from src.models.torch_deepspeed_cnn import TorchDeepSpeedCNN, TorchCNNConfig

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    batch_sizes: List[int] = None
    input_shapes: List[Tuple[int, ...]] = None
    num_warmup_runs: int = 10
    num_benchmark_runs: int = 100
    device: str = "cpu"
    save_results: bool = True
    results_dir: str = "./benchmarks/results"
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16, 32, 64]
        if self.input_shapes is None:
            self.input_shapes = [(224, 224, 3), (32, 32, 3)]

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    model_name: str
    framework: str
    batch_size: int
    input_shape: Tuple[int, ...]
    mean_latency: float
    std_latency: float
    min_latency: float
    max_latency: float
    throughput: float
    memory_usage: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result_dict = asdict(self)
        result_dict['input_shape'] = str(self.input_shape)
        return result_dict

class ModelBenchmark:
    """Benchmarking class for deep learning models."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        
        # Create results directory if it doesn't exist
        if self.config.save_results:
            os.makedirs(self.config.results_dir, exist_ok=True)
    
    def benchmark_flax_model(self, model_name: str, input_shape: Tuple[int, ...]) -> List[BenchmarkResult]:
        """Benchmark a Flax model."""
        results = []
        
        # Create model configuration
        if "MLP" in model_name:
            config = FlaxMLPConfig(
                num_classes=1000,
                hidden_sizes=[512, 256, 128],
                dropout_rate=0.1
            )
            model = FlaxMLP(config=config)
            # Flatten input for MLP
            flat_input_shape = (np.prod(input_shape),)
        else:
            config = FlaxCNNConfig(
                num_classes=1000,
                dropout_rate=0.1
            )
            model = FlaxCNN(config=config)
            flat_input_shape = input_shape
        
        # Benchmark different batch sizes
        for batch_size in self.config.batch_sizes:
            print(f"Benchmarking {model_name} (Flax) with batch size {batch_size}...")
            
            # Create input data
            input_data = jnp.ones((batch_size, *flat_input_shape), dtype=jnp.float32)
            
            # Initialize parameters
            variables = model.init(jax.random.PRNGKey(0), input_data)
            
            # Warmup runs
            for _ in range(self.config.num_warmup_runs):
                _ = model.apply(variables, input_data)
            
            # Benchmark runs
            latencies = []
            for _ in range(self.config.num_benchmark_runs):
                start_time = time.perf_counter()
                _ = model.apply(variables, input_data)
                end_time = time.perf_counter()
                latencies.append(end_time - start_time)
            
            # Calculate statistics
            mean_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)
            throughput = batch_size / mean_latency
            
            # Memory usage estimation (simplified)
            memory_usage = batch_size * np.prod(input_shape) * 4  # 4 bytes per float32
            
            # Create result
            result = BenchmarkResult(
                model_name=model_name,
                framework="Flax",
                batch_size=batch_size,
                input_shape=input_shape,
                mean_latency=mean_latency,
                std_latency=std_latency,
                min_latency=min_latency,
                max_latency=max_latency,
                throughput=throughput,
                memory_usage=memory_usage,
                timestamp=datetime.now().isoformat()
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def benchmark_torch_model(self, model_name: str, input_shape: Tuple[int, ...]) -> List[BenchmarkResult]:
        """Benchmark a PyTorch model."""
        results = []
        
        # Create model configuration
        if "MLP" in model_name:
            config = TorchMLPConfig(
                num_classes=1000,
                hidden_sizes=[512, 256, 128],
                dropout_rate=0.1
            )
            model = TorchDeepSpeedMLP(config=config)
            # Flatten input for MLP
            flat_input_shape = (np.prod(input_shape),)
        else:
            config = TorchCNNConfig(
                num_classes=1000,
                dropout_rate=0.1
            )
            model = TorchDeepSpeedCNN(config=config)
            flat_input_shape = input_shape
        
        # Move model to device
        if self.config.device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
            device = "cuda"
        else:
            device = "cpu"
        
        # Benchmark different batch sizes
        for batch_size in self.config.batch_sizes:
            print(f"Benchmarking {model_name} (PyTorch) with batch size {batch_size}...")
            
            # Create input data
            input_data = torch.ones((batch_size, *flat_input_shape), dtype=torch.float32)
            if device == "cuda":
                input_data = input_data.cuda()
            
            # Warmup runs
            for _ in range(self.config.num_warmup_runs):
                _ = model(input_data)
            
            # Benchmark runs
            latencies = []
            for _ in range(self.config.num_benchmark_runs):
                start_time = time.perf_counter()
                _ = model(input_data)
                end_time = time.perf_counter()
                latencies.append(end_time - start_time)
            
            # Calculate statistics
            mean_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)
            throughput = batch_size / mean_latency
            
            # Memory usage estimation (simplified)
            memory_usage = batch_size * np.prod(input_shape) * 4  # 4 bytes per float32
            
            # Create result
            result = BenchmarkResult(
                model_name=model_name,
                framework="PyTorch",
                batch_size=batch_size,
                input_shape=input_shape,
                mean_latency=mean_latency,
                std_latency=std_latency,
                min_latency=min_latency,
                max_latency=max_latency,
                throughput=throughput,
                memory_usage=memory_usage,
                timestamp=datetime.now().isoformat()
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        print("Starting benchmarks...")
        
        # Benchmark Flax models
        self.benchmark_flax_model("FlaxMLP", (224, 224, 3))
        self.benchmark_flax_model("FlaxCNN", (224, 224, 3))
        
        # Benchmark PyTorch models
        self.benchmark_torch_model("TorchMLP", (224, 224, 3))
        self.benchmark_torch_model("TorchCNN", (224, 224, 3))
        
        print("Benchmarks completed!")
        return self.results
    
    def save_results(self, filename: str = None):
        """Save benchmark results to file."""
        if not self.config.save_results or not self.results:
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}"
        
        # Save as JSON
        json_path = os.path.join(self.config.results_dir, f"{filename}.json")
        results_dict = [result.to_dict() for result in self.results]
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save as CSV
        csv_path = os.path.join(self.config.results_dir, f"{filename}.csv")
        df = pd.DataFrame([result.to_dict() for result in self.results])
        df.to_csv(csv_path, index=False)
        
        print(f"Results saved to {json_path} and {csv_path}")
    
    def print_summary(self):
        """Print a summary of the benchmark results."""
        if not self.results:
            print("No results to summarize.")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        # Group results by model and framework
        grouped_results = {}
        for result in self.results:
            key = f"{result.model_name} ({result.framework})"
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Print summary for each model
        for model_key, results in grouped_results.items():
            print(f"\n{model_key}:")
            print("-" * len(model_key))
            
            # Sort by batch size for consistent output
            results.sort(key=lambda x: x.batch_size)
            
            for result in results:
                print(f"  Batch Size {result.batch_size:2d}: "
                      f"Latency={result.mean_latency*1000:6.2f}ms "
                      f"(±{result.std_latency*1000:5.2f}ms) | "
                      f"Throughput={result.throughput:8.2f} samples/sec")

def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark Deep Learning Models")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16, 32, 64],
        help="Batch sizes to benchmark"
    )
    parser.add_argument(
        "--input-shapes",
        type=str,
        nargs="+",
        default=["(224,224,3)", "(32,32,3)"],
        help="Input shapes to benchmark (as strings)"
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=10,
        help="Number of warmup runs"
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=100,
        help="Number of benchmark runs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run benchmarks on"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./benchmarks/results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Parse input shapes
    input_shapes = []
    for shape_str in args.input_shapes:
        # Remove parentheses and split by comma
        shape_str = shape_str.strip("()")
        shape_tuple = tuple(int(x) for x in shape_str.split(","))
        input_shapes.append(shape_tuple)
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        batch_sizes=args.batch_sizes,
        input_shapes=input_shapes,
        num_warmup_runs=args.warmup_runs,
        num_benchmark_runs=args.benchmark_runs,
        device=args.device,
        save_results=not args.no_save,
        results_dir=args.results_dir
    )
    
    # Create benchmark instance
    benchmark = ModelBenchmark(config)
    
    # Run benchmarks
    try:
        results = benchmark.run_benchmarks()
        
        # Print summary
        benchmark.print_summary()
        
        # Save results
        benchmark.save_results()
        
        print(f"\nBenchmarking completed successfully!")
        print(f"Results saved to {args.results_dir}")
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        raise

if __name__ == "__main__":
    main()