# Benchmark report generation script for Deep Learning HPC DEMO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import os
from typing import Dict, List, Any
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

class BenchmarkReportGenerator:
    """Generate comprehensive reports from benchmark results."""
    
    def __init__(self, results_dir: str = "./benchmarks/results"):
        self.results_dir = results_dir
        self.report_dir = os.path.join(results_dir, "reports")
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_latest_results(self) -> pd.DataFrame:
        """Load the latest benchmark results."""
        # Find the latest CSV file
        csv_files = [f for f in os.listdir(self.results_dir) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV benchmark results found")
        
        # Sort by modification time and get the latest
        csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.results_dir, x)), reverse=True)
        latest_csv = csv_files[0]
        
        # Load the data
        csv_path = os.path.join(self.results_dir, latest_csv)
        df = pd.read_csv(csv_path)
        
        # Convert input_shape string back to tuple
        df['input_shape'] = df['input_shape'].apply(lambda x: tuple(map(int, x.strip("()").split(","))))
        
        print(f"Loaded results from {latest_csv}")
        return df
    
    def generate_performance_comparison(self, df: pd.DataFrame) -> Dict[str, plt.Figure]:
        """Generate performance comparison plots."""
        figures = {}
        
        # 1. Latency comparison by model and batch size
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by model and batch size
        grouped = df.groupby(['model_name', 'framework', 'batch_size'])['mean_latency'].mean().reset_index()
        grouped['model_framework'] = grouped['model_name'] + ' (' + grouped['framework'] + ')'
        
        # Pivot for heatmap
        pivot_df = grouped.pivot(index='batch_size', columns='model_framework', values='mean_latency')
        
        # Plot heatmap
        sns.heatmap(pivot_df * 1000, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_title('Mean Latency (ms) by Model and Batch Size')
        ax.set_xlabel('Model (Framework)')
        ax.set_ylabel('Batch Size')
        
        figures['latency_heatmap'] = fig
        
        # 2. Throughput comparison
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot throughput by model and batch size
        for (model_name, framework), group in df.groupby(['model_name', 'framework']):
            label = f"{model_name} ({framework})"
            ax.plot(group['batch_size'], group['throughput'], marker='o', label=label, linewidth=2)
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Throughput (samples/sec)')
        ax.set_title('Throughput Comparison')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        figures['throughput_comparison'] = fig
        
        # 3. Latency vs Throughput scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot each model-framework combination
        for (model_name, framework), group in df.groupby(['model_name', 'framework']):
            label = f"{model_name} ({framework})"
            ax.scatter(group['mean_latency'] * 1000, group['throughput'], label=label, s=100, alpha=0.7)
        
        ax.set_xlabel('Mean Latency (ms)')
        ax.set_ylabel('Throughput (samples/sec)')
        ax.set_title('Latency vs Throughput Trade-off')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        figures['latency_throughput_tradeoff'] = fig
        
        return figures
    
    def generate_detailed_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed statistical analysis."""
        stats = {}
        
        # Overall statistics
        stats['total_benchmarks'] = len(df)
        stats['models_tested'] = df[['model_name', 'framework']].drop_duplicates().shape[0]
        stats['batch_sizes_tested'] = sorted(df['batch_size'].unique())
        
        # Performance statistics by model
        model_stats = {}
        for (model_name, framework), group in df.groupby(['model_name', 'framework']):
            key = f"{model_name} ({framework})"
            model_stats[key] = {
                'mean_latency_ms': group['mean_latency'].mean() * 1000,
                'std_latency_ms': group['std_latency'].mean() * 1000,
                'min_latency_ms': group['min_latency'].min() * 1000,
                'max_latency_ms': group['max_latency'].max() * 1000,
                'mean_throughput': group['throughput'].mean(),
                'max_throughput': group['throughput'].max(),
                'min_throughput': group['throughput'].min()
            }
        
        stats['model_statistics'] = model_stats
        
        # Best performers
        best_latency_idx = df['mean_latency'].idxmin()
        best_throughput_idx = df['throughput'].idxmax()
        
        stats['best_latency'] = {
            'model': f"{df.loc[best_latency_idx, 'model_name']} ({df.loc[best_latency_idx, 'framework']})",
            'batch_size': df.loc[best_latency_idx, 'batch_size'],
            'latency_ms': df.loc[best_latency_idx, 'mean_latency'] * 1000
        }
        
        stats['best_throughput'] = {
            'model': f"{df.loc[best_throughput_idx, 'model_name']} ({df.loc[best_throughput_idx, 'framework']})",
            'batch_size': df.loc[best_throughput_idx, 'batch_size'],
            'throughput': df.loc[best_throughput_idx, 'throughput']
        }
        
        return stats
    
    def generate_framework_comparison(self, df: pd.DataFrame) -> Dict[str, plt.Figure]:
        """Generate framework-specific comparisons."""
        figures = {}
        
        # 1. Framework latency comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Box plot of latencies by framework
        sns.boxplot(data=df, x='framework', y='mean_latency', ax=ax)
        ax.set_ylabel('Mean Latency (seconds)')
        ax.set_title('Latency Distribution by Framework')
        
        # Convert to milliseconds for better readability
        ax.set_yticklabels([f'{y*1000:.2f}' for y in ax.get_yticks()])
        ax.set_ylabel('Mean Latency (milliseconds)')
        
        figures['framework_latency'] = fig
        
        # 2. Framework throughput comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Box plot of throughput by framework
        sns.boxplot(data=df, x='framework', y='throughput', ax=ax)
        ax.set_ylabel('Throughput (samples/sec)')
        ax.set_title('Throughput Distribution by Framework')
        
        figures['framework_throughput'] = fig
        
        return figures
    
    def generate_html_report(self, df: pd.DataFrame, stats: Dict[str, Any], 
                           performance_figs: Dict[str, plt.Figure],
                           framework_figs: Dict[str, plt.Figure]) -> str:
        """Generate HTML report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Deep Learning HPC DEMO - Benchmark Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }}
        .stat-label {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 8px;
        }}
        .model-stats {{
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Deep Learning HPC DEMO - Benchmark Report</h1>
        <p>Generated on: {timestamp}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats['total_benchmarks']}</div>
                <div class="stat-label">Total Benchmarks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['models_tested']}</div>
                <div class="stat-label">Models Tested</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(stats['batch_sizes_tested'])}</div>
                <div class="stat-label">Batch Sizes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{max([s['max_throughput'] for s in stats['model_statistics'].values()]):,.0f}</div>
                <div class="stat-label">Max Throughput (samples/sec)</div>
            </div>
        </div>
        
        <h3>Best Performers</h3>
        <ul>
            <li><strong>Lowest Latency:</strong> {stats['best_latency']['model']} at batch size {stats['best_latency']['batch_size']} ({stats['best_latency']['latency_ms']:.2f} ms)</li>
            <li><strong>Highest Throughput:</strong> {stats['best_throughput']['model']} at batch size {stats['best_throughput']['batch_size']} ({stats['best_throughput']['throughput']:,.0f} samples/sec)</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Performance Comparison</h2>
        <img src="latency_heatmap.png" alt="Latency Heatmap">
        <img src="throughput_comparison.png" alt="Throughput Comparison">
        <img src="latency_throughput_tradeoff.png" alt="Latency vs Throughput Trade-off">
    </div>
    
    <div class="section">
        <h2>Framework Comparison</h2>
        <img src="framework_latency.png" alt="Framework Latency">
        <img src="framework_throughput.png" alt="Framework Throughput">
    </div>
    
    <div class="section">
        <h2>Detailed Model Statistics</h2>
        """
        
        # Add model statistics
        for model_name, model_stats in stats['model_statistics'].items():
            html_content += f"""
        <div class="model-stats">
            <h3>{model_name}</h3>
            <table>
                <tr>
                    <td>Average Latency</td>
                    <td>{model_stats['mean_latency_ms']:.2f} ms</td>
                </tr>
                <tr>
                    <td>Latency Std Dev</td>
                    <td>{model_stats['std_latency_ms']:.2f} ms</td>
                </tr>
                <tr>
                    <td>Min Latency</td>
                    <td>{model_stats['min_latency_ms']:.2f} ms</td>
                </tr>
                <tr>
                    <td>Max Latency</td>
                    <td>{model_stats['max_latency_ms']:.2f} ms</td>
                </tr>
                <tr>
                    <td>Average Throughput</td>
                    <td>{model_stats['mean_throughput']:,.0f} samples/sec</td>
                </tr>
                <tr>
                    <td>Max Throughput</td>
                    <td>{model_stats['max_throughput']:,.0f} samples/sec</td>
                </tr>
            </table>
        </div>
            """
        
        html_content += """
    </div>
    
    <div class="section">
        <h2>Raw Benchmark Data</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Framework</th>
                    <th>Batch Size</th>
                    <th>Mean Latency (ms)</th>
                    <th>Throughput (samples/sec)</th>
                </tr>
            </thead>
            <tbody>
        """
        
        # Add raw data table
        for _, row in df.iterrows():
            html_content += f"""
                <tr>
                    <td>{row['model_name']}</td>
                    <td>{row['framework']}</td>
                    <td>{row['batch_size']}</td>
                    <td>{row['mean_latency'] * 1000:.2f}</td>
                    <td>{row['throughput']:,.0f}</td>
                </tr>
            """
        
        html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def save_figures(self, figures: Dict[str, plt.Figure]) -> List[str]:
        """Save figures to files."""
        saved_files = []
        
        for name, fig in figures.items():
            filename = f"{name}.png"
            filepath = os.path.join(self.report_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(filename)
            print(f"Saved figure: {filename}")
        
        return saved_files
    
    def generate_report(self):
        """Generate complete benchmark report."""
        print("Generating benchmark report...")
        
        try:
            # Load latest results
            df = self.load_latest_results()
            
            # Generate performance comparison figures
            performance_figs = self.generate_performance_comparison(df)
            
            # Generate framework comparison figures
            framework_figs = self.generate_framework_comparison(df)
            
            # Generate detailed statistics
            stats = self.generate_detailed_statistics(df)
            
            # Save all figures
            all_figures = {**performance_figs, **framework_figs}
            saved_files = self.save_figures(all_figures)
            
            # Generate HTML report
            html_content = self.generate_html_report(df, stats, performance_figs, framework_figs)
            
            # Save HTML report
            html_path = os.path.join(self.report_dir, "benchmark_report.html")
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            print(f"Report generated successfully!")
            print(f"HTML report saved to: {html_path}")
            print(f"Figures saved to: {self.report_dir}")
            
            # Print summary statistics
            print("\n" + "="*50)
            print("REPORT SUMMARY")
            print("="*50)
            print(f"Total benchmarks: {stats['total_benchmarks']}")
            print(f"Models tested: {stats['models_tested']}")
            print(f"Batch sizes tested: {stats['batch_sizes_tested']}")
            print(f"Best latency: {stats['best_latency']['latency_ms']:.2f} ms "
                  f"({stats['best_latency']['model']}, batch size {stats['best_latency']['batch_size']})")
            print(f"Best throughput: {stats['best_throughput']['throughput']:,.0f} samples/sec "
                  f"({stats['best_throughput']['model']}, batch size {stats['best_throughput']['batch_size']})")
            
        except Exception as e:
            print(f"Error generating report: {e}")
            raise

def main():
    """Main function to generate benchmark report."""
    parser = argparse.ArgumentParser(description="Generate Benchmark Report")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./benchmarks/results",
        help="Directory containing benchmark results"
    )
    
    args = parser.parse_args()
    
    # Create report generator
    generator = BenchmarkReportGenerator(results_dir=args.results_dir)
    
    # Generate report
    generator.generate_report()

if __name__ == "__main__":
    main()