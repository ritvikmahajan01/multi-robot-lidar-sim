import numpy as np
import time
from datetime import datetime
from mapping_utils import RobotOccupancyGrid, combine_grid_maps, visualize_all_maps
import os
import csv


def main():
    """Example usage of the RobotOccupancyGrid class."""
    # Load data
    data = np.load('robot_data_large.npy', allow_pickle=True).item()

    
    # Count number of LiDAR readings in dataset
    robot1_readings = sum(len(readings) for readings in data['robot1']['lidar_readings'])
    robot2_readings = sum(len(readings) for readings in data['robot2']['lidar_readings'])
    total_readings = robot1_readings + robot2_readings
    
    # Create and update occupancy grid for robot 1
    start_time1 = time.time()
    grid1 = RobotOccupancyGrid(resolution=0.05)
    grid1.initialize_grid(data)
    poses1 = data['robot1']['poses']
    lidar_readings1 = data['robot1']['lidar_readings']
    for pose, readings in zip(poses1, lidar_readings1):
        grid1.update_from_lidar(pose, readings, 'robot1')
    occupancy1 = grid1.get_occupancy_grid()
    end_time1 = time.time()
    
    # Create and update occupancy grid for robot 2
    start_time2 = time.time()
    grid2 = RobotOccupancyGrid(resolution=0.05)
    grid2.initialize_grid(data)
    poses2 = data['robot2']['poses']
    lidar_readings2 = data['robot2']['lidar_readings']
    for pose, readings in zip(poses2, lidar_readings2):
        grid2.update_from_lidar(pose, readings, 'robot2')
    occupancy2 = grid2.get_occupancy_grid()
    end_time2 = time.time()
    
    # Combine maps
    start_time_combined = time.time()
    combined_map = combine_grid_maps(grid1, grid2, data)
    end_time_combined = time.time()
    
    # Calculate total readings (LiDAR + free points)
    robot1_total = robot1_readings + grid1.free_points_sampled
    robot2_total = robot2_readings + grid2.free_points_sampled
    total_points = robot1_total + robot2_total
    
    # Save timing and point count information to CSV
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare data for CSV
    csv_data = {
        'timestamp': timestamp,
        'robot1_time': end_time1 - start_time1,
        'robot2_time': end_time2 - start_time2,
        'combined_time': end_time_combined - start_time_combined,
        'robot1_readings': robot1_readings,
        'robot2_readings': robot2_readings,
        'total_readings': total_readings,
        'robot1_total_points': robot1_total,
        'robot2_total_points': robot2_total,
        'total_points': total_points
    }
    
    # Write to CSV file
    csv_file = 'data/times_combine_ogm.csv'
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_data)
    
    # Print timing information
    print(f"\nRobot 1 processing time: {end_time1 - start_time1:.2f} seconds")
    print(f"Robot 2 processing time: {end_time2 - start_time2:.2f} seconds")
    print(f"Combined map processing time: {end_time_combined - start_time_combined:.2f} seconds")
    
    # Print reading counts
    print(f"\nRobot 1 LiDAR readings: {robot1_readings}")
    print(f"Robot 2 LiDAR readings: {robot2_readings}")
    print(f"Total LiDAR readings: {total_readings}")
    
    # Print total points (LiDAR + free)
    print(f"\nRobot 1 total points (LiDAR + free): {robot1_total}")
    print(f"Robot 2 total points (LiDAR + free): {robot2_total}")
    print(f"Total points (LiDAR + free): {total_points}")
    
    # Visualize the maps
    visualize_all_maps(grid1, grid2, combined_map, data)

if __name__ == "__main__":
    main()