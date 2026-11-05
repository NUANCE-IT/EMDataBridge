#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark runner script for electron microscopy tools.

This script provides examples of how to use the benchmarking module
to evaluate the performance of the electron microscopy tools.
"""

import os
from pathlib import Path
from EMDataBridge.benchmark import EMBenchmark


def run_sample_benchmarks():
    """Run a sample benchmark suite."""
    # Create benchmark directory
    benchmark_dir = Path("benchmark_results")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Initialize benchmark tool
    benchmark = EMBenchmark(output_dir=benchmark_dir)
    
    # Sample data directory - update this path to your test data location
    sample_data_dir = Path("sample_data")
    
    # Check if sample data directory exists
    if not sample_data_dir.exists():
        print(f"Sample data directory {sample_data_dir} not found.")
        print("Please update the path to your test data location.")
        return
    
    # Find sample files for different formats
    sample_files = [
        # Add paths to your sample files here
        # Example:
        # sample_data_dir / "sample.dm4",
        # sample_data_dir / "sample.tif",
    ]
    
    # Find all files in the sample directory if no specific files are listed
    if not sample_files:
        sample_files = list(sample_data_dir.glob("*.*"))
        # Filter out non-data files
        sample_files = [f for f in sample_files if f.suffix.lower() not in [".py", ".txt", ".md"]]
    
    if not sample_files:
        print("No sample files found. Please add sample files to the sample_data directory.")
        return
    
    print(f"Found {len(sample_files)} sample files for benchmarking.")
    for file in sample_files:
        print(f"  - {file}")
    
    # Define output formats to test
    output_formats = ["tiff", "hdf5", "mrc", "msa"]
    
    # Add Thermo Fisher Scientific formats
    thermo_fisher_formats = ["emd", "ser"]
    output_formats.extend(thermo_fisher_formats)
    
    # Add Oxford Instruments formats
    oxford_formats = ["oip", "inca", "azw", "azd"]
    output_formats.extend(oxford_formats)
    
    # Add Hitachi format
    output_formats.append("hds")
    
    print(f"\nTesting {len(output_formats)} output formats: {', '.join(output_formats)}")
    
    # Run standardization benchmark
    print("\n1. Benchmarking standardization...")
    std_results = benchmark.benchmark_standardization(sample_files)
    
    # Run translation benchmark
    print("\n2. Benchmarking format translation...")
    trans_results = benchmark.benchmark_translation(sample_files, output_formats)
    
    # Run batch processing benchmark
    print("\n3. Benchmarking batch processing...")
    batch_results = benchmark.benchmark_batch_processing(sample_data_dir, output_formats)
    
    # Generate report
    print("\nGenerating benchmark report...")
    benchmark.generate_report()
    
    print("\nBenchmark complete! Results saved to benchmark_results directory.")
    print("Check the visualizations folder for performance charts.")


def main():
    """Main function."""
    print("Electron Microscopy Tools - Benchmark Suite")
    print("-" * 50)
    
    run_sample_benchmarks()
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
