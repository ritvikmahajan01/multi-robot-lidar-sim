import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import time
from mapping_utils import RobotOccupancyGrid, combine_grid_maps, evaluate_map
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from env_utils import ROBOT_CONFIG, ENVIRONMENT_CONFIG
import math
from typing import Dict, Tuple
import random

def add_noise_to_readings(readings, range_std=0.4, angle_std=0.0):
    """Add noise to LiDAR readings."""
    noisy_readings = []
    for reading_list in readings:
        noisy_reading_list = []
        for reading in reading_list:
            # Convert tuple to numpy array
            reading_array = np.array(reading)
            if range_std > 0:
                # Add range noise and ensure it's positive
                noisy_range = reading_array[0] + np.random.normal(0, range_std)
                reading_array[0] = max(0.0, noisy_range)  # Ensure range is not negative
            if angle_std > 0:
                # Add angle noise
                reading_array[1] += np.random.normal(0, angle_std)
            noisy_reading_list.append(reading_array)
        noisy_readings.append(noisy_reading_list)
    return noisy_readings

def get_fraction_data(data, step):
    """
    Get a fraction of the data by taking every nth measurement.
    
    Args:
        data: Full robot data dictionary
        fraction: Fraction of data to use (e.g., 0.5 for 50%)
    
    Returns:
        Dictionary with fraction of data
    """

    # For each robot, take every nth pose and lidar reading
    fraction_data = {
            'poses': data['poses'][::step],
            'lidar_readings': data['lidar_readings'][::step]
        }
        

    
    return fraction_data




def count_total_points(data):
    """Count total LiDAR readings in the dataset."""
    robot1_readings = sum(len(readings) for readings in data['robot1']['lidar_readings'])
    robot2_readings = sum(len(readings) for readings in data['robot2']['lidar_readings'])
    return robot1_readings + robot2_readings

def evaluate_map_with_timing(data, ground_truth):
    """
    Evaluate map quality and measure timing for individual robots and map merging.
    
    Returns:
        Dictionary with timing and metrics
    """
    # Count total points in this fraction
    total_readings = count_total_points(data)
    
    # Create and update occupancy grid for robot 1
    start_time1 = time.time()
    grid1 = RobotOccupancyGrid(resolution=0.05)
    grid1.initialize_grid(data)
    poses1 = data['robot1']['poses']
    lidar_readings1 = data['robot1']['lidar_readings']
    for pose, readings in zip(poses1, lidar_readings1):
        grid1.update_from_lidar(pose, readings, 'robot1')
    end_time1 = time.time()
    
    # Create and update occupancy grid for robot 2
    start_time2 = time.time()
    grid2 = RobotOccupancyGrid(resolution=0.05)
    grid2.initialize_grid(data)
    poses2 = data['robot2']['poses']
    lidar_readings2 = data['robot2']['lidar_readings']
    for pose, readings in zip(poses2, lidar_readings2):
        grid2.update_from_lidar(pose, readings, 'robot2')
    end_time2 = time.time()
    
    # Combine maps
    start_time_combined = time.time()
    combined_map = combine_grid_maps(grid1, grid2, data)
    end_time_combined = time.time()
    
    # Get observation regions
    region1 = grid1.get_observation_region('robot1', data)
    region2 = grid2.get_observation_region('robot2', data)
    combined_region = region1 | region2
    
    # Evaluate each map
    metrics1 = evaluate_map(grid1.get_occupancy_grid(), ground_truth, region1)
    metrics2 = evaluate_map(grid2.get_occupancy_grid(), ground_truth, region2)
    metrics_combined = evaluate_map(combined_map, ground_truth, combined_region)
    
    # Calculate total points (LiDAR + free points)
    robot1_total = sum(len(readings) for readings in data['robot1']['lidar_readings']) + grid1.free_points_sampled
    robot2_total = sum(len(readings) for readings in data['robot2']['lidar_readings']) + grid2.free_points_sampled
    total_points = robot1_total + robot2_total
    
    return {
        'robot1_time': end_time1 - start_time1,
        'robot2_time': end_time2 - start_time2,
        'combined_time': end_time_combined - start_time_combined,
        'total_time': (end_time1 - start_time1) + (end_time2 - start_time2) + (end_time_combined - start_time_combined),
        'total_readings': total_readings,
        'total_points': total_points,
        'robot1_roc_auc': metrics1['roc_auc'],
        'robot1_nll': metrics1['nll'],
        'robot1_unknown_percentage': metrics1['unknown_percentage'],
        'robot1_classified_area': metrics1['classified_area_percentage'],
        'robot1_total_points': robot1_total,
        'robot2_roc_auc': metrics2['roc_auc'],
        'robot2_nll': metrics2['nll'],
        'robot2_unknown_percentage': metrics2['unknown_percentage'],
        'robot2_classified_area': metrics2['classified_area_percentage'],
        'robot2_total_points': robot2_total,
        'combined_roc_auc': metrics_combined['roc_auc'],
        'combined_nll': metrics_combined['nll'],
        'combined_unknown_percentage': metrics_combined['unknown_percentage'],
        'combined_classified_area': metrics_combined['classified_area_percentage']
    }

def analyze_time_performance():
    """Analyze time performance and map quality vs data fraction."""
    # Load data
    data = np.load('robot_data.npy', allow_pickle=True).item()
    ground_truth = np.load('ground_truth_mid.npy', allow_pickle=True)
    

    steps = [1, 2, 3, 4, 5, 10, 20, 50, 85, 120, 200]
    # steps = [int(1/fraction) for fraction in fractions]
    

    
    # Initialize results list
    results = []
    
    print("Starting time performance analysis...")
    print(f"Testing {len(steps)} different data steps...")
    
    for step in steps:
        print(f"\nTesting step: {step}")
        
        # Get fraction of data
        fraction_data_robot1 = get_fraction_data(data['robot1'], step)
        fraction_data_robot2 = get_fraction_data(data['robot2'], step)
        
        # Add noise to the fraction data
        noisy_data = {
            'robot1': {
                'poses': fraction_data_robot1['poses'],
                'lidar_readings': add_noise_to_readings(fraction_data_robot1['lidar_readings'], range_std=0.5)
            },
            'robot2': {
                'poses': fraction_data_robot2['poses'],
                'lidar_readings': add_noise_to_readings(fraction_data_robot2['lidar_readings'], range_std=0.5)
            }
        }
        
        # Evaluate with timing
        result = evaluate_map_with_timing(noisy_data, ground_truth)
        
        # Add metadata
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result['step'] = step
        result['step_percentage'] = (1.0 / step) * 100
        # result['total_points'] = count_total_points(noisy_data)
        
        results.append(result)
        
        # Print progress
        print(f"  Total points: {result['total_points']}")
        print(f"  Robot 1 time: {result['robot1_time']:.3f}s")
        print(f"  Robot 2 time: {result['robot2_time']:.3f}s")
        print(f"  Robot 1 total points: {result['robot1_total_points']}")
        print(f"  Robot 2 total points: {result['robot2_total_points']}")
        print(f"  Combined time: {result['combined_time']:.3f}s")
        print(f"  Combined ROC AUC: {result['combined_roc_auc']:.3f}")
        print(f"  Combined NLL: {result['combined_nll']:.3f}")
        print(f"  Combined Unknown %: {result['combined_unknown_percentage']:.1f}%")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save results
    csv_path = 'data/time_analysis_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create plots
    plot_results_with_points(df)
    
    return df

def plot_results_with_points(df):
    """Plot results from the DataFrame."""

    points_in_thousands = df['total_points'] / 1000

    # Plot 1: Timing vs Fraction
    plt.figure()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    plt.plot(points_in_thousands, df['robot1_time'], 'o-', label='Robot 1', color='red')
    plt.plot(points_in_thousands, df['robot2_time'], 's-', label='Robot 2', color='blue')
    plt.plot(points_in_thousands, df['combined_time'], '^-', label='Map Merging', color='green')
    plt.plot(points_in_thousands, df['total_time'], 'd-', label='Total Time', color='black')
    plt.xlabel('Total Points (thousands)')
    plt.ylabel('Time (seconds)')
    plt.title('Processing Time vs Total Points')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Plot 2: ROC AUC vs Fraction
    plt.figure()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    plt.plot(points_in_thousands, df['robot1_roc_auc'], 'o-', label='Robot 1', color='red')
    plt.plot(points_in_thousands, df['robot2_roc_auc'], 's-', label='Robot 2', color='blue')
    plt.plot(points_in_thousands, df['combined_roc_auc'], '^-', label='Combined', color='green')
    plt.xlabel('Total Points (thousands)')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC vs Total Points')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Plot 3: NLL vs Fraction
    plt.figure()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    plt.plot(points_in_thousands, df['robot1_nll'], 'o-', label='Robot 1', color='red')
    plt.plot(points_in_thousands, df['robot2_nll'], 's-', label='Robot 2', color='blue')
    plt.plot(points_in_thousands, df['combined_nll'], '^-', label='Combined', color='green')
    plt.xlabel('Total Points (thousands)')
    plt.ylabel('Negative Log Likelihood')
    plt.title('NLL vs Total Points')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Plot 5: Time per Point vs Fraction
    time_per_point = df['total_time'] / df['total_readings']
    plt.figure()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    plt.plot(points_in_thousands, time_per_point, 'o-', color='purple')
    plt.xlabel('Total Points (thousands)')
    plt.ylabel('Time per reading (seconds)')
    plt.title('Processing Efficiency vs Total Points')
    plt.grid(True)
    plt.xticks(rotation=45)

    # Plot 6: Classified Area vs Fraction
    plt.figure()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    plt.plot(points_in_thousands, df['robot1_classified_area'], 'o-', label='Robot 1', color='red')
    plt.plot(points_in_thousands, df['robot2_classified_area'], 's-', label='Robot 2', color='blue')
    plt.plot(points_in_thousands, df['combined_classified_area'], '^-', label='Combined', color='green')
    plt.xlabel('Total Points (thousands)')
    plt.ylabel('Classified Area (%)')
    plt.title('Classified Area vs Total Points')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.show()
    
    # print("Plots saved to: data/time_analysis_plots.png")

def analyze_noise_performance():
    """Analyze time performance and map quality vs noise, keeping step size constant."""

    noise_values = [0.1,0.2,0.3,0.4,0.5] # Example: 0.0, 0.1, ..., 0.6

    # Load data
    data = np.load('robot_data.npy', allow_pickle=True).item()
    ground_truth = np.load('ground_truth_mid.npy', allow_pickle=True)

    # Get fraction of data for the fixed step
    fraction_data_robot1 = get_fraction_data(data['robot1'], 85)
    fraction_data_robot2 = get_fraction_data(data['robot2'], 120)

    results = []
    print("Starting noise performance analysis...")
    # print(f"Testing {len(noise_values)} different noise values at step size {step}...")

    for noise in noise_values:
        print(f"\nTesting noise: {noise}")
        # Add noise to the fraction data
        noisy_data = {
            'robot1': {
                'poses': fraction_data_robot1['poses'],
                'lidar_readings': add_noise_to_readings(fraction_data_robot1['lidar_readings'], range_std=noise)
            },
            'robot2': {
                'poses': fraction_data_robot2['poses'],
                'lidar_readings': add_noise_to_readings(fraction_data_robot2['lidar_readings'], range_std=noise)
            }
        }

        # Evaluate with timing
        result = evaluate_map_with_timing(noisy_data, ground_truth)
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # result['step'] = step
        result['noise'] = noise
        results.append(result)

        # Print progress
        print(f"  Total points: {result['total_points']}")
        print(f"  Robot 1 time: {result['robot1_time']:.3f}s")
        print(f"  Robot 2 time: {result['robot2_time']:.3f}s")
        print(f"  Robot 1 total points: {result['robot1_total_points']}")
        print(f"  Robot 2 total points: {result['robot2_total_points']}")
        print(f"  Combined time: {result['combined_time']:.3f}s")
        print(f"  Combined ROC AUC: {result['combined_roc_auc']:.3f}")
        print(f"  Combined NLL: {result['combined_nll']:.3f}")
        print(f"  Combined Unknown %: {result['combined_unknown_percentage']:.1f}%")

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    os.makedirs('data', exist_ok=True)
    csv_path = f'data/noise_analysis_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    plot_results_with_noise(df)
    return df

def plot_results_with_noise(df):
    """Plot results from the DataFrame with noise on the x-axis."""
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    x = df['noise']

    # Plot 1: Timing vs Noise
    plt.figure()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    plt.plot(x, df['robot1_time'], 'o-', label='Robot 1', color='red')
    plt.plot(x, df['robot2_time'], 's-', label='Robot 2', color='blue')
    plt.plot(x, df['combined_time'], '^-', label='Map Merging', color='green')
    plt.plot(x, df['total_time'], 'd-', label='Total Time', color='black')
    plt.xlabel('Noise (std)')
    plt.ylabel('Time (seconds)')
    plt.title('Processing Time vs Noise')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Plot 2: ROC AUC vs Noise
    plt.figure()
    plt.plot(x, df['robot1_roc_auc'], 'o-', label='Robot 1', color='red')
    plt.plot(x, df['robot2_roc_auc'], 's-', label='Robot 2', color='blue')
    plt.plot(x, df['combined_roc_auc'], '^-', label='Combined', color='green')
    plt.xlabel('Noise (std)')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC vs Noise')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Plot 3: NLL vs Noise
    plt.figure()
    plt.plot(x, df['robot1_nll'], 'o-', label='Robot 1', color='red')
    plt.plot(x, df['robot2_nll'], 's-', label='Robot 2', color='blue')
    plt.plot(x, df['combined_nll'], '^-', label='Combined', color='green')
    plt.xlabel('Noise (std)')
    plt.ylabel('Negative Log Likelihood')
    plt.title('NLL vs Noise')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Plot 4: Classified Area vs Noise
    plt.figure()
    plt.plot(x, df['robot1_classified_area'], 'o-', label='Robot 1', color='red')
    plt.plot(x, df['robot2_classified_area'], 's-', label='Robot 2', color='blue')
    plt.plot(x, df['combined_classified_area'], '^-', label='Combined', color='green')
    plt.xlabel('Noise (std)')
    plt.ylabel('Classified Area (%)')
    plt.title('Classified Area vs Noise')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Plot 5: Time per Point vs Noise
    # time_per_point = df['total_time'] / df['total_readings']
    # plt.figure()
    # plt.plot(x, time_per_point, 'o-', color='purple')
    # plt.xlabel('Noise (std)')
    # plt.ylabel('Time per reading (seconds)')
    # plt.title('Processing Efficiency vs Noise')
    # plt.grid(True)
    # plt.xticks(rotation=45)

    plt.show()

def main():
    """Main function to run the time analysis."""
    print("Starting Time Performance Analysis")
    print("=" * 50)
    
    # Run the analysis
    # results_df = analyze_time_performance()
    results_df = analyze_noise_performance()


if __name__ == "__main__":
    main() 