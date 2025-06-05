import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import math


def load_robot_data(filename: str) -> Dict:
    """Load robot data from .npy file."""
    data = np.load(filename, allow_pickle=True).item()
    return data

def sample_free_points(robot_pose: Tuple[float, float, float], 
                      lidar_readings: List[Tuple[float, float]], 
                      num_points: int = 5) -> List[Tuple[float, float]]:
    """
    Sample free points along LiDAR beams.
    
    Args:
        robot_pose: (x, y, theta) tuple of robot position and orientation
        lidar_readings: List of (range, bearing) tuples from LiDAR
        num_points: Number of points to sample along each beam
        
    Returns:
        List of (x, y) coordinates of free points
    """
    free_points = []
    robot_x, robot_y, robot_theta = robot_pose
    
    for range_val, bearing in lidar_readings:
        # Skip readings at max range (no obstacle detected)
        if range_val >= 3.0:  # Maximum LiDAR range is 3.0m
            continue
            
        # Calculate end point of LiDAR ray
        bearing_rad = math.radians(bearing)
        end_x = robot_x + range_val * math.cos(robot_theta + bearing_rad)
        end_y = robot_y + range_val * math.sin(robot_theta + bearing_rad)
        
        # Sample points along the ray
        for i in range(num_points):
            t = (i + 1) / (num_points + 1)  # Skip the robot position (t=0)
            x = robot_x + t * (end_x - robot_x)
            y = robot_y + t * (end_y - robot_y)
            free_points.append((x, y))
            
    return free_points

def create_point_dataset(data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a dataset of free and occupied points from robot data.
    
    Args:
        data: Dictionary containing robot poses and LiDAR readings
        
    Returns:
        Tuple of (free_points, occupied_points, free_robot_ids, occupied_robot_ids) where:
        - free_points: Nx2 array of (x,y) coordinates of free points
        - occupied_points: Mx2 array of (x,y) coordinates of occupied points
        - free_robot_ids: Array indicating which robot sampled each free point (1 or 2)
        - occupied_robot_ids: Array indicating which robot detected each occupied point (1 or 2)
    """
    all_free_points = []
    all_occupied_points = []
    all_free_robot_ids = []
    all_occupied_robot_ids = []
    
    for robot_id in ['robot1', 'robot2']:
        robot_num = 1 if robot_id == 'robot1' else 2
        poses = data[robot_id]['poses']
        lidar_readings = data[robot_id]['lidar_readings']
        
        for pose, readings in zip(poses, lidar_readings):
            # Get free points
            free_points = sample_free_points(pose, readings)
            all_free_points.extend(free_points)
            all_free_robot_ids.extend([robot_num] * len(free_points))
            
            # Get occupied points (end points of LiDAR rays)
            robot_x, robot_y, robot_theta = pose
            for range_val, bearing in readings:
                if range_val >= 3.0:  # Skip max range readings
                    continue
                bearing_rad = math.radians(bearing)
                end_x = robot_x + range_val * math.cos(robot_theta + bearing_rad)
                end_y = robot_y + range_val * math.sin(robot_theta + bearing_rad)
                all_occupied_points.append((end_x, end_y))
                all_occupied_robot_ids.append(robot_num)
    
    return (np.array(all_free_points), 
            np.array(all_occupied_points), 
            np.array(all_free_robot_ids),
            np.array(all_occupied_robot_ids))

def visualize_point_dataset(free_points: np.ndarray, 
                          occupied_points: np.ndarray, 
                          free_robot_ids: np.ndarray,
                          occupied_robot_ids: np.ndarray,
                          data: Dict) -> None:
    """
    Visualize the free and occupied points dataset along with robot trajectories.
    
    Args:
        free_points: Nx2 array of (x,y) coordinates of free points
        occupied_points: Mx2 array of (x,y) coordinates of occupied points
        free_robot_ids: Array indicating which robot sampled each free point (1 or 2)
        occupied_robot_ids: Array indicating which robot detected each occupied point (1 or 2)
        data: Dictionary containing robot poses and LiDAR readings
    """
    plt.figure(figsize=(10, 8))
    
    # Plot free points with different colors for each robot
    robot1_free_mask = free_robot_ids == 1
    robot2_free_mask = free_robot_ids == 2
    
    plt.scatter(free_points[robot1_free_mask, 0], free_points[robot1_free_mask, 1],
               c='pink', alpha=0.3, label='Robot 1 Free Space')
    plt.scatter(free_points[robot2_free_mask, 0], free_points[robot2_free_mask, 1],
               c='lightblue', alpha=0.3, label='Robot 2 Free Space')
    
    # Plot occupied points with different colors for each robot
    robot1_occ_mask = occupied_robot_ids == 1
    robot2_occ_mask = occupied_robot_ids == 2
    
    plt.scatter(occupied_points[robot1_occ_mask, 0], occupied_points[robot1_occ_mask, 1],
               c='red', alpha=0.7, label='Robot 1 Detections')
    plt.scatter(occupied_points[robot2_occ_mask, 0], occupied_points[robot2_occ_mask, 1],
               c='blue', alpha=0.7, label='Robot 2 Detections')
    
    # Plot robot trajectories
    robot1_poses = np.array(data['robot1']['poses'])
    robot2_poses = np.array(data['robot2']['poses'])
    
    plt.plot(robot1_poses[:, 0], robot1_poses[:, 1], 'r-', linewidth=2, label='Robot 1 Trajectory')
    plt.plot(robot2_poses[:, 0], robot2_poses[:, 1], 'b-', linewidth=2, label='Robot 2 Trajectory')
    
    # Plot start positions
    plt.scatter(robot1_poses[0, 0], robot1_poses[0, 1], c='red', s=100, marker='^', label='Robot 1 Start')
    plt.scatter(robot2_poses[0, 0], robot2_poses[0, 1], c='blue', s=100, marker='^', label='Robot 2 Start')
    
    plt.title('Free and Occupied Points Dataset with Robot Trajectories')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def main():
    # Load the data
    data = load_robot_data('robot_data.npy')
    
    # Create point dataset
    free_points, occupied_points, free_robot_ids, occupied_robot_ids = create_point_dataset(data)
    
    # Print dataset statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Number of free points:")
    print(f"  - Robot 1: {np.sum(free_robot_ids == 1)}")
    print(f"  - Robot 2: {np.sum(free_robot_ids == 2)}")
    print(f"Number of occupied points:")
    print(f"  - Robot 1: {np.sum(occupied_robot_ids == 1)}")
    print(f"  - Robot 2: {np.sum(occupied_robot_ids == 2)}")
    
    # Visualize the dataset
    visualize_point_dataset(free_points, occupied_points, free_robot_ids, occupied_robot_ids, data)

if __name__ == "__main__":
    main()