import numpy as np
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from mapping_utils import RobotOccupancyGrid, combine_grid_maps, evaluate_map
from env_utils import ROBOT_CONFIG
import time
import matplotlib.pyplot as plt

def add_noise_to_readings(readings, range_std=0.1, angle_std=0.0):
    """Add Gaussian noise to LiDAR readings."""
    noisy_readings = []
    for reading_list in readings:
        noisy_reading_list = []
        for reading in reading_list:
            reading_array = np.array(reading)
            if range_std > 0:
                noisy_range = reading_array[0] + np.random.normal(0, range_std)
                reading_array[0] = max(0.0, noisy_range)
            if angle_std > 0:
                reading_array[1] += np.random.normal(0, angle_std)
            noisy_reading_list.append(tuple(reading_array))
        noisy_readings.append(noisy_reading_list)
    return noisy_readings

def add_noise_to_poses(poses, pos_std=0.05, theta_std=0.01):
    """Add Gaussian noise to robot poses (x, y, theta)."""
    noisy_poses = []
    for pose in poses:
        x, y, theta = pose
        noisy_x = x + np.random.normal(0, pos_std)
        noisy_y = y + np.random.normal(0, pos_std)
        noisy_theta = theta + np.random.normal(0, theta_std)
        noisy_poses.append((noisy_x, noisy_y, noisy_theta))
    return noisy_poses

def create_point_dataset_robot(data: Dict, free_per_beam: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a dataset of free and occupied points from robot data.
    Args:
        data: Dictionary containing robot poses and LiDAR readings
    Returns:
        Tuple of (free_points, occupied_points) where:
        - free_points: Nx2 array of (x,y) coordinates of free points
        - occupied_points: Mx2 array of (x,y) coordinates of occupied points
    """
    all_free_points = []
    all_occupied_points = []
    poses = data['poses']
    lidar_readings = data['lidar_readings']
    for pose, readings in zip(poses, lidar_readings):
        robot_x, robot_y, robot_theta = pose
        free_points = []
        for range_val, bearing in readings:
            bearing_rad = np.radians(bearing)
            if range_val >= ROBOT_CONFIG['lidar_range']:
                end_x = robot_x + ROBOT_CONFIG['lidar_range'] * np.cos(robot_theta + bearing_rad)
                end_y = robot_y + ROBOT_CONFIG['lidar_range'] * np.sin(robot_theta + bearing_rad)
            else:
                end_x = robot_x + range_val * np.cos(robot_theta + bearing_rad)
                end_y = robot_y + range_val * np.sin(robot_theta + bearing_rad)
                all_occupied_points.append((end_x, end_y))
            for i in range(free_per_beam):
                t = (i + 1) / (free_per_beam + 1)
                x = robot_x + t * (end_x - robot_x)
                y = robot_y + t * (end_y - robot_y)
                free_points.append((x, y))
        all_free_points.extend(free_points)
    return (np.array(all_free_points), np.array(all_occupied_points))

def create_point_dataset_all(data: dict = np.load('robot_data.npy', allow_pickle=True).item(), free_per_beam: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a dataset of free and occupied points for both robots from already noisy data.
    """
    all_free_points = {"robot1": [], "robot2": []}
    all_occupied_points = {"robot1": [], "robot2": []}
    for robot in ['robot1', 'robot2']:
        free_points, occupied_points = create_point_dataset_robot(data[robot], free_per_beam=free_per_beam)
        all_free_points[robot].extend(free_points)
        all_occupied_points[robot].extend(occupied_points)

    # print(f"Number of free points: {len(all_free_points['robot1'])}")
    # print(f"Number of occupied points: {len(all_occupied_points['robot1'])}")
    # print(f"Number of free points: {len(all_free_points['robot2'])}")
    # print(f"Number of occupied points: {len(all_occupied_points['robot2'])}")

    return (
        np.array(all_free_points["robot1"]),
        np.array(all_free_points["robot2"]),
        np.array(all_occupied_points["robot1"]),
        np.array(all_occupied_points["robot2"])
    )

def update_grid_from_points(grid, free_points, occupied_points, observation_region):
    grid.initialize_grid()
    # Free points
    for x, y in free_points:
        grid_x, grid_y = grid.world_to_grid(x, y)
        if 0 <= grid_x < grid.grid.shape[1] and 0 <= grid_y < grid.grid.shape[0]:
            observation_region[grid_y, grid_x] = True
            grid.log_odds[grid_y, grid_x] = np.clip(
                grid.log_odds[grid_y, grid_x] + grid.l_free,
                -100, 100
            )
    # Occupied points
    for x, y in occupied_points:
        grid_x, grid_y = grid.world_to_grid(x, y)
        if 0 <= grid_x < grid.grid.shape[1] and 0 <= grid_y < grid.grid.shape[0]:
            observation_region[grid_y, grid_x] = True
            grid.log_odds[grid_y, grid_x] = np.clip(
                grid.log_odds[grid_y, grid_x] + grid.l_occ,
                -100, 100
            )

    # Convert to occupancy grid
    # grid.get_occupancy_grid()

def combine_ogm_batch(grid1: RobotOccupancyGrid, grid2: RobotOccupancyGrid, observation_region1: np.ndarray, observation_region2: np.ndarray) -> np.ndarray:
    """Combine maps from two robots by adding log-odds."""
    # Ensure boolean type for logical operations
    region1 = observation_region1.astype(bool)
    region2 = observation_region2.astype(bool)
    # Initialize combined log-odds map
    combined_log_odds = np.zeros_like(grid1.log_odds)
    # Update regions observed by only one robot
    only_region1 = region1 & ~region2
    only_region2 = region2 & ~region1
    combined_log_odds[only_region1] = grid1.log_odds[only_region1]
    combined_log_odds[only_region2] = grid2.log_odds[only_region2]
    # Update overlapping regions by adding log-odds
    overlap = region1 & region2
    if np.any(overlap):
        combined_log_odds[overlap] = grid1.log_odds[overlap] + grid2.log_odds[overlap]
    # Convert log-odds to probabilities
    return 1 / (1 + np.exp(-combined_log_odds))

def process_ogm_and_combine(
    noisy_data: dict,
    num_sampled_free: int = 10000,
    num_sampled_occupied: int = 10000,
    free_per_beam: int = 20
) -> tuple:
    free_points_robot1, free_points_robot2, occupied_points_robot1, occupied_points_robot2 = create_point_dataset_all(noisy_data, free_per_beam=free_per_beam)
    n_free = min(num_sampled_free, len(free_points_robot1))
    if num_sampled_free > len(free_points_robot1):
        print(f"Warning: num_sampled_free is greater than the number of free points for robot 1: {num_sampled_free} > {len(free_points_robot1)}")
    n_occ = min(num_sampled_occupied, len(occupied_points_robot1))
    if num_sampled_occupied > len(occupied_points_robot1):
        print(f"Warning: num_sampled_occupied is greater than the number of occupied points for robot 1: {num_sampled_occupied} > {len(occupied_points_robot1)}")
    free_indices = np.random.choice(len(free_points_robot1), n_free, replace=False) if n_free > 0 else []
    occ_indices = np.random.choice(len(occupied_points_robot1), n_occ, replace=False) if n_occ > 0 else []
    sampled_points_robot1 = {
        'free': free_points_robot1[free_indices] if n_free > 0 else np.empty((0,2)),
        'occupied': occupied_points_robot1[occ_indices] if n_occ > 0 else np.empty((0,2))
    }
    n_free = min(num_sampled_free, len(free_points_robot2))
    n_occ = min(num_sampled_occupied, len(occupied_points_robot2))
    if num_sampled_occupied > len(occupied_points_robot2):
        print(f"Warning: num_sampled_occupied is greater than the number of occupied points for robot 2: {num_sampled_occupied} > {len(occupied_points_robot2)}")
    if num_sampled_free > len(free_points_robot2):
        print(f"Warning: num_sampled_free is greater than the number of free points for robot 2: {num_sampled_free} > {len(free_points_robot2)}")
    free_indices = np.random.choice(len(free_points_robot2), n_free, replace=False) if n_free > 0 else []
    occ_indices = np.random.choice(len(occupied_points_robot2), n_occ, replace=False) if n_occ > 0 else []
    sampled_points_robot2 = {
        'free': free_points_robot2[free_indices] if n_free > 0 else np.empty((0,2)),
        'occupied': occupied_points_robot2[occ_indices] if n_occ > 0 else np.empty((0,2))
    }
    # Mapping for robot 1
    start_time = time.time()
    grid1 = RobotOccupancyGrid(resolution=0.05)
    grid1.initialize_grid()
    observation_region1 = np.zeros_like(grid1.log_odds, dtype=bool)
    update_grid_from_points(grid1, sampled_points_robot1['free'], sampled_points_robot1['occupied'], observation_region1)
    robot1_time = time.time() - start_time
    # Mapping for robot 2
    start_time = time.time()
    grid2 = RobotOccupancyGrid(resolution=0.05)
    grid2.initialize_grid()
    observation_region2 = np.zeros_like(grid2.log_odds, dtype=bool)
    update_grid_from_points(grid2, sampled_points_robot2['free'], sampled_points_robot2['occupied'], observation_region2)
    robot2_time = time.time() - start_time
    # Combine maps
    start_time = time.time()
    combined_map = combine_ogm_batch(grid1, grid2, observation_region1, observation_region2)
    combined_time = time.time() - start_time

    # Plot the combined and individual maps
    # plt.figure(figsize=(10, 6))
    # plt.imshow(combined_map, cmap='viridis')
    # plt.colorbar()
    # plt.title('Combined Map')


    # plt.figure(figsize=(10, 6))
    # plt.imshow(grid1.get_occupancy_grid(), cmap='viridis')
    # plt.colorbar()
    # plt.title('Robot 1 Map')

    # plt.figure(figsize=(10, 6))
    # plt.imshow(grid2.get_occupancy_grid(), cmap='viridis')
    # plt.colorbar()
    # plt.title('Robot 2 Map')
    # plt.show()


    return grid1, observation_region1, grid2, observation_region2, combined_map, robot1_time, robot2_time, combined_time

def test_metrics_batch(
    robot_data_path='robot_data.npy',
    ground_truth_path='ground_truth_mid.npy',
    pose_noise_std=0.0,
    theta_noise_std=0.0,
    angle_noise_std=0.0,
    range_noise_std=0.0,
    out_csv='data/noise_analysis_results.csv',
    num_sampled_free=1000000,
    num_sampled_occupied=50000,
    free_per_beam=20
):
    """Test metrics by varying LiDAR range noise, using sampled points from noisy data."""
    data = np.load(robot_data_path, allow_pickle=True).item()
    ground_truth = np.load(ground_truth_path, allow_pickle=True)
    results = []
    noisy_data = {
        'robot1': {
            'poses': add_noise_to_poses(data['robot1']['poses'], pos_std=pose_noise_std, theta_std=theta_noise_std),
            'lidar_readings': add_noise_to_readings(data['robot1']['lidar_readings'], range_std=range_noise_std, angle_std=angle_noise_std)
        },
        'robot2': {
            'poses': add_noise_to_poses(data['robot2']['poses'], pos_std=pose_noise_std, theta_std=theta_noise_std),
            'lidar_readings': add_noise_to_readings(data['robot2']['lidar_readings'], range_std=range_noise_std, angle_std=angle_noise_std)
        }
    }
    grid1, observation_region1, grid2, observation_region2, combined_map, robot1_time, robot2_time, combined_time = process_ogm_and_combine(
        noisy_data,
        num_sampled_free=num_sampled_free,
        num_sampled_occupied=num_sampled_occupied,
        free_per_beam=free_per_beam
    )
    metrics1 = evaluate_map(grid1.get_occupancy_grid(), ground_truth, observation_region1)
    metrics2 = evaluate_map(grid2.get_occupancy_grid(), ground_truth, observation_region2)
    metrics_combined = evaluate_map(combined_map, ground_truth, observation_region1 | observation_region2)
    result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'range_noise_std': range_noise_std,
        'angle_noise_std': angle_noise_std,
        'pose_noise_std': pose_noise_std,
        'theta_noise_std': theta_noise_std,
        'num_sampled_free': num_sampled_free,
        'num_sampled_occupied': num_sampled_occupied,
        'free_per_beam': free_per_beam,
        'robot1_time': robot1_time,
        'robot2_time': robot2_time,
        'combined_time': combined_time,
        'robot1_roc_auc': metrics1['roc_auc'],
        'robot2_roc_auc': metrics2['roc_auc'],
        'combined_roc_auc': metrics_combined['roc_auc'],
        'robot1_nll': metrics1['nll'],
        'robot2_nll': metrics2['nll'],
        'combined_nll': metrics_combined['nll'],
        'robot1_unknown_percentage': metrics1['unknown_percentage'],
        'robot2_unknown_percentage': metrics2['unknown_percentage'],
        'combined_unknown_percentage': metrics_combined['unknown_percentage'],
        'robot1_classified_area': metrics1['classified_area_percentage'],
        'robot2_classified_area': metrics2['classified_area_percentage'],
        'combined_classified_area': metrics_combined['classified_area_percentage']
    }
    results.append(result)
    print(f"Range noise {range_noise_std}: Combined ROC AUC {metrics_combined['roc_auc']:.3f}, NLL {metrics_combined['nll']:.3f}")
    df = pd.DataFrame(results)
    # os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    # df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")
    return df

def analyze_performance_with_noise():
    range_variance_values = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
    range_std_values = np.sqrt(range_variance_values)
    angle_variance = 0.0
    angle_std = np.sqrt(angle_variance)
    pose_variance = 0.0
    pose_std = np.sqrt(pose_variance)
    theta_variance = 0.00
    theta_std = np.sqrt(theta_variance)

    all_metrics = []
    for range_std in range_std_values:
        print(f"Processing range variance: {range_std}")
        metrics = test_metrics_batch(
            range_noise_std=range_std,
            angle_noise_std=angle_std,
            pose_noise_std=pose_std,
            theta_noise_std=theta_std
        )
        # metrics is a DataFrame with one row, so take the first row as dict
        if isinstance(metrics, pd.DataFrame):
            all_metrics.append(metrics.iloc[0].to_dict())
        else:
            all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)
    # Save to csv
    df.to_csv('data/noise_analysis_results.csv', index=False)

    # Plot ROC AUC
    plt.figure(figsize=(10, 6))
    plt.plot(range_variance_values, df['robot1_roc_auc'], label='Robot 1 ROC AUC')
    plt.plot(range_variance_values, df['robot2_roc_auc'], label='Robot 2 ROC AUC')
    plt.plot(range_variance_values, df['combined_roc_auc'], label='Combined ROC AUC')
    plt.legend()
    plt.title('ROC AUC')
    plt.savefig('data/roc_auc.png')

    # Plot NLL
    plt.figure(figsize=(10, 6))
    plt.plot(range_variance_values, df['robot1_nll'], label='Robot 1 NLL')
    plt.plot(range_variance_values, df['robot2_nll'], label='Robot 2 NLL')
    plt.plot(range_variance_values, df['combined_nll'], label='Combined NLL')
    plt.legend()
    plt.title('NLL')
    plt.savefig('data/nll.png')

    # Plot unknown percentage
    plt.figure(figsize=(10, 6))
    plt.plot(range_variance_values, df['robot1_unknown_percentage'], label='Robot 1 Unknown Percentage')
    plt.plot(range_variance_values, df['robot2_unknown_percentage'], label='Robot 2 Unknown Percentage')
    plt.plot(range_variance_values, df['combined_unknown_percentage'], label='Combined Unknown Percentage')
    plt.legend()
    plt.title('Unknown Percentage')
    plt.savefig('data/unknown_percentage.png')  

    # Plot classified area percentage
    plt.figure(figsize=(10, 6))
    plt.plot(range_variance_values, df['robot1_classified_area'], label='Robot 1 Classified Area Percentage')
    plt.plot(range_variance_values, df['robot2_classified_area'], label='Robot 2 Classified Area Percentage')
    plt.plot(range_variance_values, df['combined_classified_area'], label='Combined Classified Area Percentage')
    plt.legend()
    plt.title('Classified Area Percentage')
    plt.savefig('data/classified_area_percentage.png')

    # Plot times
    plt.figure(figsize=(10, 6))
    plt.plot(range_variance_values, df['robot1_time'], label='Robot 1 Time')
    plt.plot(range_variance_values, df['robot2_time'], label='Robot 2 Time')
    plt.plot(range_variance_values, df['combined_time'], label='Combined Time')
    plt.legend()
    plt.title('Times')
    plt.savefig('data/times.png')
    plt.show()

def analyze_performance_with_size():
    num_sampled = [10000, 50000, 100000, 500000, 1000000]

    all_metrics = []
    for num_of_points in num_sampled:
        print(f"Processing num_of_points: {num_of_points}")
        metrics = test_metrics_batch(
            num_sampled_free=num_of_points,
            num_sampled_occupied=num_of_points
        )
        if isinstance(metrics, pd.DataFrame):
            all_metrics.append(metrics.iloc[0].to_dict())
        else:
            all_metrics.append(metrics)

    df = pd.DataFrame(all_metrics)

    # Save to csv
    df.to_csv('data/size_analysis_results.csv', index=False)

    # Plot ROC AUC
    plt.figure(figsize=(10, 6))

if __name__ == "__main__":
    # analyze_performance_with_noise()
    analyze_performance_with_size()
    # create_point_dataset_all()
