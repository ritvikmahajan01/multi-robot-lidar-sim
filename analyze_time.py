import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import time
from mapping_utils import RobotOccupancyGrid, combine_grid_maps, evaluate_map

def add_noise_to_readings(readings, range_std=0.1, angle_std=0.0):
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
    
    fraction_data = {
        'robot1': {
            'poses': data['robot1']['poses'][::step],
            'lidar_readings': data['robot1']['lidar_readings'][::step]
        },
        'robot2': {
            'poses': data['robot2']['poses'][::step],
            'lidar_readings': data['robot2']['lidar_readings'][::step]
        }
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
        'robot1_unknown_percentage': metrics1['unknown_percentage'] * 100,
        'robot2_roc_auc': metrics2['roc_auc'],
        'robot2_nll': metrics2['nll'],
        'robot2_unknown_percentage': metrics2['unknown_percentage'] * 100,
        'combined_roc_auc': metrics_combined['roc_auc'],
        'combined_nll': metrics_combined['nll'],
        'combined_unknown_percentage': metrics_combined['unknown_percentage'] * 100
    }

def analyze_time_performance():
    """Analyze time performance and map quality vs data fraction."""
    # Load data
    data = np.load('robot_data_large.npy', allow_pickle=True).item()
    ground_truth = np.load('ground_truth_large.npy', allow_pickle=True)
    
    # Define fractions to test (every nth measurement)
    steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    

    
    # Initialize results list
    results = []
    
    print("Starting time performance analysis...")
    print(f"Testing {len(steps)} different data fractions...")
    
    for step in steps:
        print(f"\nTesting step: {step}")
        
        # Get fraction of data
        fraction_data = get_fraction_data(data, step)
        
        # Add noise to the fraction data
        noisy_data = {
            'robot1': {
                'poses': fraction_data['robot1']['poses'],
                'lidar_readings': add_noise_to_readings(fraction_data['robot1']['lidar_readings'], range_std=0.1)
            },
            'robot2': {
                'poses': fraction_data['robot2']['poses'],
                'lidar_readings': add_noise_to_readings(fraction_data['robot2']['lidar_readings'], range_std=0.1)
            }
        }
        
        # Evaluate with timing
        result = evaluate_map_with_timing(noisy_data, ground_truth)
        
        # Add metadata
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result['step'] = step
        result['fraction_percentage'] = (1.0 / step) * 100
        
        results.append(result)
        
        # Print progress
        print(f"  Total points: {result['total_points']}")
        print(f"  Robot 1 time: {result['robot1_time']:.3f}s")
        print(f"  Robot 2 time: {result['robot2_time']:.3f}s")
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
    plot_results(df)
    
    return df

def plot_results(df):
    """Plot results from the DataFrame."""

    
    # Plot 1: Timing vs Fraction
    plt.figure()
    plt.plot(df['fraction_percentage'], df['robot1_time'], 'o-', label='Robot 1', color='red')
    plt.plot(df['fraction_percentage'], df['robot2_time'], 's-', label='Robot 2', color='blue')
    plt.plot(df['fraction_percentage'], df['combined_time'], '^-', label='Map Merging', color='green')
    plt.plot(df['fraction_percentage'], df['total_time'], 'd-', label='Total Time', color='black')
    plt.xlabel('Fraction of Data Considered')
    plt.ylabel('Time (seconds)')
    plt.title('Processing Time vs Fraction of Data Considered')
    plt.legend()
    plt.grid(True)
    

    # Plot 2: ROC AUC vs Fraction
    plt.figure()
    plt.plot(df['fraction_percentage'], df['robot1_roc_auc'], 'o-', label='Robot 1', color='red')
    plt.plot(df['fraction_percentage'], df['robot2_roc_auc'], 's-', label='Robot 2', color='blue')
    plt.plot(df['fraction_percentage'], df['combined_roc_auc'], '^-', label='Combined', color='green')
    plt.xlabel('Fraction of Data Considered')
    plt.ylabel('ROC AUC')
    plt.title('ROC AUC vs Fraction of Data Considered')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: NLL vs Fraction
    plt.figure()
    plt.plot(df['fraction_percentage'], df['robot1_nll'], 'o-', label='Robot 1', color='red')
    plt.plot(df['fraction_percentage'], df['robot2_nll'], 's-', label='Robot 2', color='blue')
    plt.plot(df['fraction_percentage'], df['combined_nll'], '^-', label='Combined', color='green')
    plt.xlabel('Fraction of Data Considered')
    plt.ylabel('Negative Log Likelihood')
    plt.title('NLL vs Fraction of Data Considered')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Unknown Percentage vs Fraction
    plt.figure()
    plt.plot(df['fraction_percentage'], df['robot1_unknown_percentage'], 'o-', label='Robot 1', color='red')
    plt.plot(df['fraction_percentage'], df['robot2_unknown_percentage'], 's-', label='Robot 2', color='blue')
    plt.plot(df['fraction_percentage'], df['combined_unknown_percentage'], '^-', label='Combined', color='green')
    plt.xlabel('Fraction of Data Considered')
    plt.ylabel('Unknown Percentage (%)')
    plt.title('Unknown Percentage vs Fraction of Data Considered')
    plt.legend()
    plt.grid(True)
    
    # Plot 5: Time per Point vs Fraction
    time_per_point = df['total_time'] / df['total_readings']
    plt.figure()
    plt.plot(df['fraction_percentage'], time_per_point, 'o-', color='purple')
    plt.xlabel('Fraction of Data Considered')
    plt.ylabel('Time per Point (seconds)')
    plt.title('Processing Efficiency vs Fraction of Data Considered')
    plt.grid(True)
    
    # Plot 6: Total Points vs Fraction
    # plt.figure()
    # plt.plot(df['fraction_percentage'], df['total_readings'], 'o-', color='orange')
    # plt.xlabel('Fraction of Data Considered')
    # plt.ylabel('Total Readings')
    # plt.title('Total Readings vs Fraction of Data Considered')
    # plt.grid(True)
    
    # plt.tight_layout()
    # plt.savefig('data/time_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # print("Plots saved to: data/time_analysis_plots.png")

def main():
    """Main function to run the time analysis."""
    print("Starting Time Performance Analysis")
    print("=" * 50)
    
    # Run the analysis
    results_df = analyze_time_performance()


if __name__ == "__main__":
    main() 