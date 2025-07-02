#!/usr/bin/env python3
"""
Results analysis and visualization script for LLM framework benchmarks.
Generates detailed comparisons and charts for vLLM vs Mojo/MAX performance.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Any
import statistics
from datetime import datetime

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class BenchmarkAnalyzer:
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.results = []
        self.df = None
        
    def load_results(self):
        """Load benchmark results from JSON file."""
        try:
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            
            if not self.results:
                raise ValueError("No results found in file")
                
            self.df = pd.DataFrame(self.results)
            print(f"Loaded {len(self.results)} benchmark results")
            return True
            
        except FileNotFoundError:
            print(f"Error: Results file '{self.results_file}' not found")
            return False
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in '{self.results_file}'")
            return False
        except Exception as e:
            print(f"Error loading results: {e}")
            return False
    
    def print_summary_stats(self):
        """Print high-level summary statistics."""
        if self.df is None:
            print("No data loaded")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        frameworks = self.df['framework'].unique()
        
        for framework in frameworks:
            fw_data = self.df[self.df['framework'] == framework]
            
            print(f"\n{framework.upper()} Framework Results:")
            print("-" * 50)
            
            print(f"Total test scenarios: {len(fw_data)}")
            print(f"Model tested: {fw_data['model'].iloc[0]}")
            
            # Key metrics
            avg_ttft = fw_data['ttft_ms'].mean()
            avg_tps = fw_data['tokens_per_second'].mean()
            avg_throughput = fw_data['throughput_req_per_sec'].mean()
            avg_memory = fw_data['peak_memory_mb'].mean()
            avg_cpu = fw_data['avg_cpu_percent'].mean()
            
            print(f"Average Time to First Token: {avg_ttft:.2f} ms")
            print(f"Average Tokens per Second: {avg_tps:.2f}")
            print(f"Average Throughput: {avg_throughput:.2f} req/sec")
            print(f"Average Peak Memory: {avg_memory:.2f} MB")
            print(f"Average CPU Usage: {avg_cpu:.1f}%")
            
            # Performance ranges
            print(f"\nPerformance Ranges:")
            print(f"TTFT: {fw_data['ttft_ms'].min():.2f} - {fw_data['ttft_ms'].max():.2f} ms")
            print(f"TPS: {fw_data['tokens_per_second'].min():.2f} - {fw_data['tokens_per_second'].max():.2f}")
            print(f"Throughput: {fw_data['throughput_req_per_sec'].min():.2f} - {fw_data['throughput_req_per_sec'].max():.2f} req/sec")
    
    def generate_comparison_table(self):
        """Generate detailed comparison table."""
        if len(self.df['framework'].unique()) < 2:
            print("Need at least 2 frameworks for comparison")
            return
        
        print("\n" + "="*100)
        print("DETAILED FRAMEWORK COMPARISON")
        print("="*100)
        
        # Group by test scenario characteristics
        comparison_groups = self.df.groupby(['prompt_length', 'output_length', 'concurrent_requests'])
        
        for (prompt_len, output_len, concurrent), group in comparison_groups:
            print(f"\nScenario: {prompt_len} token prompts → {output_len} tokens, {concurrent} concurrent request(s)")
            print("-" * 90)
            
            comparison_data = []
            for framework in group['framework'].unique():
                fw_data = group[group['framework'] == framework]
                if len(fw_data) > 0:
                    row = fw_data.iloc[0]  # Take first (should be only) result for this scenario
                    comparison_data.append({
                        'Framework': framework,
                        'TTFT (ms)': f"{row['ttft_ms']:.2f}",
                        'TPS': f"{row['tokens_per_second']:.2f}",
                        'Throughput (req/s)': f"{row['throughput_req_per_sec']:.2f}",
                        'Memory (MB)': f"{row['peak_memory_mb']:.1f}",
                        'CPU (%)': f"{row['avg_cpu_percent']:.1f}",
                        'P95 Latency (ms)': f"{row['latency_p95_ms']:.2f}"
                    })
            
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                print(comp_df.to_string(index=False))
                
                # Calculate performance ratios if we have exactly 2 frameworks
                if len(comparison_data) == 2:
                    fw1, fw2 = comparison_data[0], comparison_data[1]
                    ttft_ratio = float(fw1['TTFT (ms)']) / float(fw2['TTFT (ms)'])
                    tps_ratio = float(fw1['TPS']) / float(fw2['TPS'])
                    throughput_ratio = float(fw1['Throughput (req/s)']) / float(fw2['Throughput (req/s)'])
                    
                    print(f"\nPerformance Ratios ({fw1['Framework']}/{fw2['Framework']}):")
                    print(f"TTFT: {ttft_ratio:.2f}x {'(faster)' if ttft_ratio < 1 else '(slower)'}")
                    print(f"TPS: {tps_ratio:.2f}x {'(faster)' if tps_ratio > 1 else '(slower)'}")
                    print(f"Throughput: {throughput_ratio:.2f}x {'(faster)' if throughput_ratio > 1 else '(slower)'}")
    
    def create_visualizations(self, output_dir: str = "benchmark_plots"):
        """Create comprehensive visualization charts."""
        if self.df is None:
            print("No data to visualize")
            return
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Time to First Token Comparison
        self._plot_ttft_comparison(output_dir)
        
        # 2. Tokens per Second Comparison
        self._plot_tps_comparison(output_dir)
        
        # 3. Throughput Comparison
        self._plot_throughput_comparison(output_dir)
        
        # 4. Resource Usage Comparison
        self._plot_resource_usage(output_dir)
        
        # 5. Latency Distribution
        self._plot_latency_distribution(output_dir)
        
        # 6. Performance vs Concurrency
        self._plot_concurrency_impact(output_dir)
        
        # 7. Overall Performance Radar Chart
        self._plot_performance_radar(output_dir)
        
        print(f"\nVisualization charts saved to '{output_dir}/' directory")
    
    def _plot_ttft_comparison(self, output_dir: str):
        """Plot Time to First Token comparison."""
        plt.figure(figsize=(14, 8))
        
        # Create scenario labels
        self.df['scenario'] = self.df.apply(lambda x: f"{x['prompt_length']}→{x['output_length']}\n({x['concurrent_requests']} req)", axis=1)
        
        sns.barplot(data=self.df, x='scenario', y='ttft_ms', hue='framework')
        plt.title('Time to First Token (TTFT) Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Test Scenario (Prompt→Output tokens, Concurrent requests)', fontsize=12)
        plt.ylabel('Time to First Token (ms)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Framework', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/ttft_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_tps_comparison(self, output_dir: str):
        """Plot Tokens per Second comparison."""
        plt.figure(figsize=(14, 8))
        
        sns.barplot(data=self.df, x='scenario', y='tokens_per_second', hue='framework')
        plt.title('Tokens per Second (TPS) Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Test Scenario (Prompt→Output tokens, Concurrent requests)', fontsize=12)
        plt.ylabel('Tokens per Second', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Framework', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/tps_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_throughput_comparison(self, output_dir: str):
        """Plot throughput comparison."""
        plt.figure(figsize=(14, 8))
        
        sns.barplot(data=self.df, x='scenario', y='throughput_req_per_sec', hue='framework')
        plt.title('Request Throughput Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Test Scenario (Prompt→Output tokens, Concurrent requests)', fontsize=12)
        plt.ylabel('Throughput (requests/sec)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Framework', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/throughput_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_resource_usage(self, output_dir: str):
        """Plot resource usage comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Memory usage
        sns.barplot(data=self.df, x='scenario', y='peak_memory_mb', hue='framework', ax=ax1)
        ax1.set_title('Peak Memory Usage', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Test Scenario', fontsize=12)
        ax1.set_ylabel('Peak Memory (MB)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # CPU usage
        sns.barplot(data=self.df, x='scenario', y='avg_cpu_percent', hue='framework', ax=ax2)
        ax2.set_title('Average CPU Usage', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Test Scenario', fontsize=12)
        ax2.set_ylabel('CPU Usage (%)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/resource_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_latency_distribution(self, output_dir: str):
        """Plot latency percentile distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        latency_cols = ['ttft_ms', 'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms']
        titles = ['Time to First Token', 'P50 Latency', 'P95 Latency', 'P99 Latency']
        
        for i, (col, title) in enumerate(zip(latency_cols, titles)):
            sns.boxplot(data=self.df, x='framework', y=col, ax=axes[i])
            axes[i].set_title(title, fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Latency (ms)', fontsize=12)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_concurrency_impact(self, output_dir: str):
        """Plot performance vs concurrency level."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Filter for same prompt/output length to show concurrency impact
        base_scenario = self.df[(self.df['prompt_length'] == 64) & (self.df['output_length'] == 128)]
        
        if len(base_scenario) > 0:
            # TPS vs Concurrency
            sns.lineplot(data=base_scenario, x='concurrent_requests', y='tokens_per_second', 
                        hue='framework', marker='o', ax=ax1)
            ax1.set_title('Tokens/sec vs Concurrency', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Concurrent Requests', fontsize=12)
            ax1.set_ylabel('Tokens per Second', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # Throughput vs Concurrency
            sns.lineplot(data=base_scenario, x='concurrent_requests', y='throughput_req_per_sec', 
                        hue='framework', marker='o', ax=ax2)
            ax2.set_title('Throughput vs Concurrency', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Concurrent Requests', fontsize=12)
            ax2.set_ylabel('Requests per Second', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/concurrency_impact.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_radar(self, output_dir: str):
        """Create radar chart for overall performance comparison."""
        frameworks = self.df['framework'].unique()
        
        if len(frameworks) < 2:
            return
        
        # Calculate normalized metrics (0-1 scale)
        metrics = ['ttft_ms', 'tokens_per_second', 'throughput_req_per_sec', 'peak_memory_mb', 'avg_cpu_percent']
        metric_labels = ['TTFT (lower=better)', 'Tokens/sec', 'Throughput', 'Memory (lower=better)', 'CPU (lower=better)']
        
        # For TTFT, memory, and CPU - lower is better, so we invert them
        invert_metrics = ['ttft_ms', 'peak_memory_mb', 'avg_cpu_percent']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for framework in frameworks:
            fw_data = self.df[self.df['framework'] == framework]
            values = []
            
            for metric in metrics:
                avg_val = fw_data[metric].mean()
                max_val = self.df[metric].max()
                min_val = self.df[metric].min()
                
                # Normalize to 0-1 scale
                if max_val != min_val:
                    normalized = (avg_val - min_val) / (max_val - min_val)
                else:
                    normalized = 0.5
                
                # Invert for metrics where lower is better
                if metric in invert_metrics:
                    normalized = 1 - normalized
                
                values.append(normalized)
            
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=framework)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Performance Comparison\n(Higher values = Better performance)', 
                    size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_detailed_report(self, output_file: str = "benchmark_report.txt"):
        """Export detailed text report."""
        with open(output_file, 'w') as f:
            f.write("LLM FRAMEWORK BENCHMARK REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results file: {self.results_file}\n")
            f.write(f"Total scenarios tested: {len(self.df)}\n\n")
            
            # Framework summary
            for framework in self.df['framework'].unique():
                fw_data = self.df[self.df['framework'] == framework]
                f.write(f"{framework.upper()} FRAMEWORK SUMMARY\n")
                f.write("-" * 30 + "\n")
                
                f.write(f"Test scenarios: {len(fw_data)}\n")
                f.write(f"Model: {fw_data['model'].iloc[0]}\n")
                f.write(f"Average TTFT: {fw_data['ttft_ms'].mean():.2f} ms\n")
                f.write(f"Average TPS: {fw_data['tokens_per_second'].mean():.2f}\n")
                f.write(f"Average Throughput: {fw_data['throughput_req_per_sec'].mean():.2f} req/s\n")
                f.write(f"Average Memory: {fw_data['peak_memory_mb'].mean():.2f} MB\n")
                f.write(f"Average CPU: {fw_data['avg_cpu_percent'].mean():.1f}%\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-" * 30 + "\n")
            for _, row in self.df.iterrows():
                f.write(f"Framework: {row['framework']}\n")
                f.write(f"Scenario: {row['prompt_length']}→{row['output_length']} tokens, {row['concurrent_requests']} concurrent\n")
                f.write(f"TTFT: {row['ttft_ms']:.2f} ms\n")
                f.write(f"TPS: {row['tokens_per_second']:.2f}\n")
                f.write(f"Throughput: {row['throughput_req_per_sec']:.2f} req/s\n")
                f.write(f"Memory: {row['peak_memory_mb']:.2f} MB\n")
                f.write(f"CPU: {row['avg_cpu_percent']:.1f}%\n")
                f.write(f"Timestamp: {row['timestamp']}\n")
                f.write("-" * 20 + "\n")
        
        print(f"Detailed report saved to '{output_file}'")


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM framework benchmark results")
    parser.add_argument("results_file", nargs='?', default="benchmark_results.json", 
                       help="JSON file containing benchmark results")
    parser.add_argument("--output-dir", default="benchmark_plots", 
                       help="Directory to save visualization plots")
    parser.add_argument("--report", default="benchmark_report.txt", 
                       help="Filename for detailed text report")
    parser.add_argument("--no-plots", action="store_true", 
                       help="Skip generating visualization plots")
    parser.add_argument("--summary-only", action="store_true", 
                       help="Only print summary, no detailed analysis")
    
    args = parser.parse_args()
    
    analyzer = BenchmarkAnalyzer(args.results_file)
    
    if not analyzer.load_results():
        return 1
    
    # Always print summary
    analyzer.print_summary_stats()
    
    if not args.summary_only:
        # Generate detailed comparison
        analyzer.generate_comparison_table()
        
        # Generate plots unless disabled
        if not args.no_plots:
            analyzer.create_visualizations(args.output_dir)
        
        # Export detailed report
        analyzer.export_detailed_report(args.report)
    
    return 0


if __name__ == "__main__":
    exit(main())