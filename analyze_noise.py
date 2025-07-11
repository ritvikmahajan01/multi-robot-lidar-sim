import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
from mapping_utils import RobotOccupancyGrid, combine_grid_maps, create_ground_truth_grid, evaluate_map

def add_noise_to_readings(readings, range_std=0.0, angle_std=0.0):
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

def add_noise_to_poses(poses, velocity_std=0.0, angular_std=0.0):
    """Add noise to robot poses."""
    # Convert list of tuples to numpy array
    noisy_poses = np.array(poses)
    
    if velocity_std > 0 or angular_std > 0:
        for i in range(1, len(poses)):
            # Calculate actual velocity direction
            dx = poses[i][0] - poses[i-1][0]
            dy = poses[i][1] - poses[i-1][1]
            velocity_angle = np.arctan2(dy, dx)
            
            # Add noise to velocity
            if velocity_std > 0:
                velocity_noise = np.random.normal(0, velocity_std)
                noisy_poses[i, 0] += velocity_noise * np.cos(velocity_angle)
                noisy_poses[i, 1] += velocity_noise * np.sin(velocity_angle)
            
            # Add noise to angular velocity
            if angular_std > 0:
                angular_noise = np.random.normal(0, angular_std)
                noisy_poses[i, 2] += angular_noise
    
    return noisy_poses

def sample_data(data, points):
    """Randomly sample a fixed number of (pose, lidar_readings) pairs for each robot."""
    import random
    sampled_data = {}
    for robot in ['robot1', 'robot2']:
        poses = data[robot]['poses']
        lidar_readings = data[robot]['lidar_readings']
        n = len(poses)
        if points > n:
            print(f"Warning: Requested more points than available in the data. Using all {n} points.")
            idxs = list(range(n))
        else:
            idxs = random.sample(range(n), points)
        # Sort indices to preserve temporal order (optional)
        idxs.sort()
        sampled_data[robot] = {
            'poses': [poses[i] for i in idxs],
            'lidar_readings': [lidar_readings[i] for i in idxs]
        }
    return sampled_data

def evaluate_map_quality(data: dict, noise_params: dict) -> dict:
    """Evaluate map quality using data with added noise."""
    # Initialize occupancy grids for both robots
    grid1 = RobotOccupancyGrid(resolution=0.05)
    grid2 = RobotOccupancyGrid(resolution=0.05)
    
    # Initialize grids with data
    grid1.initialize_grid(data)
    grid2.initialize_grid(data)
    
    # Add noise to robot 1's data
    noisy_poses1 = add_noise_to_poses(data['robot1']['poses'], 
                                    noise_params['velocity_std'], 
                                    noise_params['angular_std'])
    noisy_readings1 = add_noise_to_readings(data['robot1']['lidar_readings'],
                                          noise_params['range_std'],
                                          noise_params['angle_std'])
    
    # Add noise to robot 2's data
    noisy_poses2 = add_noise_to_poses(data['robot2']['poses'],
                                    noise_params['velocity_std'],
                                    noise_params['angular_std'])
    noisy_readings2 = add_noise_to_readings(data['robot2']['lidar_readings'],
                                          noise_params['range_std'],
                                          noise_params['angle_std'])
    
    # Update grids with noisy LiDAR readings
    for pose, readings in zip(noisy_poses1, noisy_readings1):
        grid1.update_from_lidar(pose, readings, 'robot1')
    
    for pose, readings in zip(noisy_poses2, noisy_readings2):
        grid2.update_from_lidar(pose, readings, 'robot2')
    
    # Create ground truth grid
    ground_truth = np.load('ground_truth_mid.npy', allow_pickle=True)
    
    # Get observation regions
    region1 = grid1.get_observation_region('robot1', data)
    region2 = grid2.get_observation_region('robot2', data)
    
    # Combine maps
    combined_map = combine_grid_maps(grid1, grid2, data)
    
    # Evaluate each map
    metrics1 = evaluate_map(grid1.get_occupancy_grid(), ground_truth, region1)
    metrics2 = evaluate_map(grid2.get_occupancy_grid(), ground_truth, region2)
    metrics_combined = evaluate_map(combined_map, ground_truth, region1 | region2)
    
    return {
        'robot1': metrics1,
        'robot2': metrics2,
        'combined': metrics_combined
    }

def analyze_noise_parameters():
    """Analyze how different noise levels affect map quality."""
    # Load existing data (assumed to be noise-free)
    data = np.load('robot_data.npy', allow_pickle=True).item()

    # Get fraction of data

    # total_points_r1 = len(data['robot1']['lidar_readings'])
    # total_points_r2 = len(data['robot2']['lidar_readings'])
    points = 20000/5
    data = sample_data(data, points)

    # Print total number of lidar hits
    total_points_r1 = len(data['robot1']['lidar_readings'])
    total_points_r2 = len(data['robot2']['lidar_readings'])
    print(f"Total number of lidar hits: {total_points_r1 + total_points_r2}")
    
    # Define noise ranges to test
    noise_ranges = {
        'range_std': np.linspace(0.1, 0.5, 5),    # 0 to 0.5 meters
    }
    
    # Initialize results list
    results = []
    
    # Test each noise parameter
    for noise_type, noise_values in noise_ranges.items():
        print(f"\nTesting {noise_type}...")
        for noise_value in noise_values:
            # Set up noise parameters
            noise_params = {
                'range_std': 0.0,
                'angle_std': 0.0,
                'velocity_std': 0.0,
                'angular_std': 0.0
            }
            noise_params[noise_type] = noise_value
            
            # Evaluate map quality with current noise
            metrics = evaluate_map_quality(data, noise_params)
            
            # Record results
            result = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'noise_type': noise_type,
                'noise_value': noise_value,
                'robot1_unknown_percentage': metrics['robot1']['unknown_percentage'],
                'robot1_classified_area': metrics['robot1']['classified_area_percentage'],
                'robot1_roc_auc': metrics['robot1']['roc_auc'],
                'robot1_nll': metrics['robot1']['nll'],
                'robot2_unknown_percentage': metrics['robot2']['unknown_percentage'],
                'robot2_classified_area': metrics['robot2']['classified_area_percentage'],
                'robot2_roc_auc': metrics['robot2']['roc_auc'],
                'robot2_nll': metrics['robot2']['nll'],
                'combined_unknown_percentage': metrics['combined']['unknown_percentage'],
                'combined_classified_area': metrics['combined']['classified_area_percentage'],
                'combined_roc_auc': metrics['combined']['roc_auc'],
                'combined_nll': metrics['combined']['nll'],
            }
            results.append(result)
            
            # Print progress
            print(f"  {noise_type} = {noise_value:.3f}: "
                  f"Combined ROC AUC = {metrics['combined']['roc_auc']:.2f} "
                  f"Combined NLL = {metrics['combined']['nll']:.2f}, "
                  f"Combined Unknown Percentage = {metrics['combined']['unknown_percentage']:.2f}%"
                  f"Combined Classified Area = {metrics['combined']['classified_area_percentage']:.2f}%"
                  )
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save results
    csv_path = 'data/noise_combine_ogm.csv'
    df.to_csv(csv_path, index=False)
    
    # Create plots
    plot_results(df)

def plot_results(df):
    """Plot results from the DataFrame."""
    # Plot accuracy vs noise for each noise type
    # plt.figure()
    # for noise_type in df['noise_type'].unique():
    #     noise_data = df[df['noise_type'] == noise_type]
    #     plt.plot(noise_data['noise_value'], noise_data['combined_accuracy'], 
    #             label=noise_type, marker='o')
    # plt.title('Merged Map Accuracy vs Noise')
    # plt.xlabel('Noise Standard Deviation')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.grid(True)
    
    # Plot false negatives vs noise
    # plt.figure()
    # for noise_type in df['noise_type'].unique():
    #     noise_data = df[df['noise_type'] == noise_type]
    #     plt.plot(noise_data['noise_value'], noise_data['combined_false_negatives'], 
    #             label=noise_type, marker='o')
    # plt.title('Merged Map False Negatives vs Noise')
    # plt.xlabel('Noise Standard Deviation')
    # plt.ylabel('False Negatives (%)')
    # plt.legend()
    # plt.grid(True)
    
    # # Plot false positives vs noise
    # plt.figure()    
    # for noise_type in df['noise_type'].unique():
    #     noise_data = df[df['noise_type'] == noise_type]
    #     plt.plot(noise_data['noise_value'], noise_data['combined_false_positives'], 
    #             label=noise_type, marker='o')
    # plt.title('Merged Map False Positives vs Noise')
    # plt.xlabel('Noise Standard Deviation')
    # plt.ylabel('False Positives (%)')
    # plt.legend()
    # plt.grid(True)
    
    # Plot unknown percentage vs noise
    # plt.figure()
    # for noise_type in df['noise_type'].unique():
    #     noise_data = df[df['noise_type'] == noise_type]
    #     plt.plot(noise_data['noise_value'], noise_data['combined_unknown_percentage'], 
    #             label=noise_type, marker='o')
    # plt.title('Merged Map Unknown Cells vs Noise')
    # plt.xlabel('Noise Standard Deviation')
    # plt.ylabel('Unknown Cells (%)')
    # plt.legend()
    # plt.grid(True)

    # # Plot robot and combined accuracy vs range std
    # plt.figure()
    range_data = df[df['noise_type'] == 'range_std']
    # plt.plot(range_data['noise_value'], range_data['robot1_accuracy'], label='Robot 1 OGM Accuracy', marker='o')
    # plt.plot(range_data['noise_value'], range_data['robot2_accuracy'], label='Robot 2 OGM Accuracy', marker='o')
    # plt.plot(range_data['noise_value'], range_data['combined_accuracy'], label='Merged OGM Accuracy', marker='o')
    # plt.title('Accuracy vs Range Noise')
    # plt.xlabel('Range Standard Deviation')
    # plt.ylabel('Accuracy (%)')
    # plt.legend()
    # plt.grid(True)

    # Plot robot and combined accuracy vs angle std
    # plt.figure()
    # angle_data = df[df['noise_type'] == 'angle_std']
    # plt.plot(angle_data['noise_value'], angle_data['robot1_accuracy'], label='Robot 1 OGM Accuracy', marker='o')
    # plt.plot(angle_data['noise_value'], angle_data['robot2_accuracy'], label='Robot 2 OGMAccuracy', marker='o')
    # plt.plot(angle_data['noise_value'], angle_data['combined_accuracy'], label='Merged OGM Accuracy', marker='o')
    # plt.title('Accuracy vs Angle Noise')
    # plt.xlabel('Angle Standard Deviation')
    # plt.ylabel('Accuracy (%)')
    # plt.legend()
    # plt.grid(True)

    # Plot robot and combined false negatives vs range std
    # plt.figure()
    # plt.plot(range_data['noise_value'], range_data['robot1_false_negatives'], label='Robot 1 False Negatives', marker='o')
    # plt.plot(range_data['noise_value'], range_data['robot2_false_negatives'], label='Robot 2 False Negatives', marker='o')
    # plt.plot(range_data['noise_value'], range_data['combined_false_negatives'], label='Combined False Negatives', marker='o')
    # plt.title('False Negatives vs Range Noise')
    # plt.xlabel('Range Standard Deviation')
    # plt.ylabel('False Negatives (%)')
    # plt.legend()
    # plt.grid(True)

    # Plot robot and combined unknown percentage vs range std
    plt.figure()
    plt.plot(range_data['noise_value'], range_data['robot1_unknown_percentage'], label='Robot 1 OGM Unknown Cells (%)', marker='o')
    plt.plot(range_data['noise_value'], range_data['robot2_unknown_percentage'], label='Robot 2 OGM Unknown Cells (%)', marker='o')
    plt.plot(range_data['noise_value'], range_data['combined_unknown_percentage'], label='Merged OGM Unknown Cells (%)', marker='o')
    plt.title('Unknown Cells vs Range Noise')
    plt.xlabel('Range Standard Deviation')
    plt.ylabel('Unknown Cells (%)')
    plt.legend()
    plt.grid(True)

    # Plot robot and combined ROC AUC vs range std
    plt.figure()
    plt.plot(range_data['noise_value'], range_data['robot1_roc_auc'], label='Robot 1 ROC AUC', marker='o')
    plt.plot(range_data['noise_value'], range_data['robot2_roc_auc'], label='Robot 2 ROC AUC', marker='o')
    plt.plot(range_data['noise_value'], range_data['combined_roc_auc'], label='Merged ROC AUC', marker='o')
    plt.title('ROC AUC vs Range Noise')
    plt.ylabel('ROC AUC')
    plt.legend()
    plt.xlabel('Range Standard Deviation')

    # Plot robot and combined NLL vs range std
    plt.figure()
    plt.plot(range_data['noise_value'], range_data['robot1_nll'], label='Robot 1 NLL', marker='o')
    plt.plot(range_data['noise_value'], range_data['robot2_nll'], label='Robot 2 NLL', marker='o')
    plt.plot(range_data['noise_value'], range_data['combined_nll'], label='Merged NLL', marker='o')
    plt.title('NLL vs Range Noise')
    plt.xlabel('Range Standard Deviation')

    # Plot classified area vs range std
    plt.figure()
    plt.plot(range_data['noise_value'], range_data['robot1_classified_area'], label='Robot 1 Classified Area (%)', marker='o')
    plt.plot(range_data['noise_value'], range_data['robot2_classified_area'], label='Robot 2 Classified Area (%)', marker='o')
    plt.plot(range_data['noise_value'], range_data['combined_classified_area'], label='Merged Classified Area (%)', marker='o')
    plt.title('Classified Area vs Range Noise')
    plt.xlabel('Range Standard Deviation')

    # Plot robot and combined false negatives vs angle std
    # plt.figure()
    # plt.plot(angle_data['noise_value'], angle_data['robot1_false_negatives'], label='Robot 1 False Negatives', marker='o')
    # plt.plot(angle_data['noise_value'], angle_data['robot2_false_negatives'], label='Robot 2 False Negatives', marker='o')
    # plt.plot(angle_data['noise_value'], angle_data['combined_false_negatives'], label='Combined False Negatives', marker='o')
    # plt.title('False Negatives vs Angle Noise')
    # plt.xlabel('Angle Standard Deviation')
    # plt.ylabel('False Negatives (%)')
    # plt.legend()
    # plt.grid(True)

    # Plot robot and combined unknown percentage vs angle std
    # plt.figure()
    # plt.plot(angle_data['noise_value'], angle_data['robot1_unknown_percentage'], label='Robot 1 OGM Unknown Cells (%)', marker='o')
    # plt.plot(angle_data['noise_value'], angle_data['robot2_unknown_percentage'], label='Robot 2 OGM Unknown Cells (%)', marker='o')
    # plt.plot(angle_data['noise_value'], angle_data['combined_unknown_percentage'], label='Merged OGM Unknown Cells (%)', marker='o')
    # plt.title('Unknown Cells vs Angle Noise')
    # plt.xlabel('Angle Standard Deviation')
    # plt.ylabel('Unknown Cells (%)')
    # plt.legend()
    # plt.grid(True)
    
    plt.show()

def main():
    """Main function with interactive prompt."""
    csv_path = 'data/noise_combine_ogm.csv'
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Run simulations and plot results")
        print("2. Plot results from existing data")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\nRunning simulations...")
            analyze_noise_parameters()
        elif choice == '2':
            if not os.path.exists(csv_path):
                print(f"\nError: No data file found at {csv_path}")
                continue
            print("\nLoading data from CSV file...")
            df = pd.read_csv(csv_path)
            plot_results(df)
        elif choice == '3':
            print("\nExiting...")
            break
        else:
            print("\nInvalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main() 