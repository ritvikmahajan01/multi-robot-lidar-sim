import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import math
from env_utils import ENVIRONMENT_CONFIG, ROBOT_CONFIG
import mapping_utils

def load_robot_data(filename: str) -> Dict:
    """Load robot data from .npy file."""
    data = np.load(filename, allow_pickle=True).item()
    return data

def create_occupancy_grid(data: Dict) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
    """Create an occupancy grid map from LiDAR readings."""
    # Find map boundaries
    min_x = min_y = 0.0
    max_x = ENVIRONMENT_CONFIG['x_width']
    max_y = ENVIRONMENT_CONFIG['y_height']
    resolution = mapping_utils.default_resolution

    # for robot_id in ['robot1', 'robot2']:
    #     poses = data[robot_id]['poses']
    #     for pose in poses:
    #         min_x = min(min_x, pose[0])
    #         max_x = max(max_x, pose[0])
    #         min_y = min(min_y, pose[1])
    #         max_y = max(max_y, pose[1])
    
    # Add some padding
    # padding = 1.0
    # min_x -= padding
    # min_y -= padding
    # max_x += padding
    # max_y += padding
    
    # Create grid
    grid_width = int((max_x - min_x) / resolution)
    grid_height = int((max_y - min_y) / resolution)
    occupancy_grid = np.zeros((grid_height, grid_width))
    robot_grid = np.zeros((grid_height, grid_width))  # 1 for robot1, 2 for robot2
    
    # Convert world coordinates to grid coordinates
    def world_to_grid(x: float, y: float) -> Tuple[int, int]:
        grid_x = int((x - min_x) / resolution)
        grid_y = int((y - min_y) / resolution)
        return grid_x, grid_y
    
    # Process LiDAR readings
    for robot_id in ['robot1', 'robot2']:
        poses = data[robot_id]['poses']
        lidar_readings = data[robot_id]['lidar_readings']
        robot_value = 1 if robot_id == 'robot1' else 2
        
        for pose, readings in zip(poses, lidar_readings):
            robot_x, robot_y, robot_theta = pose
            
            for range_val, bearing in readings:
                # Skip readings that are at max range (no obstacle detected)
                if range_val >= ROBOT_CONFIG['lidar_range']:  # Maximum LiDAR range is 3.0m
                    continue
                    
                # Calculate end point of LiDAR ray
                bearing_rad = math.radians(bearing)
                end_x = robot_x + range_val * math.cos(robot_theta + bearing_rad)
                end_y = robot_y + range_val * math.sin(robot_theta + bearing_rad)
                
                # Mark the end point as occupied
                grid_x, grid_y = world_to_grid(end_x, end_y)
                if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                    occupancy_grid[grid_y, grid_x] = 1
                    robot_grid[grid_y, grid_x] = robot_value
    
    return occupancy_grid, robot_grid, (min_x, max_x, min_y, max_y)

def plot_robot_trajectories_and_map(data: Dict) -> None:
    """Plot robot trajectories and occupancy grid map."""
    # Create occupancy grid
    occupancy_grid, robot_grid, (min_x, max_x, min_y, max_y) = create_occupancy_grid(data)
    
    # Create a custom colormap for the colored occupancy grid
    # 0: white (empty), 1: red (robot1), 2: blue (robot2)
    colors = ['white', 'red', 'blue']
    cmap = plt.cm.colors.ListedColormap(colors)
    
    # Plot colored occupancy grid
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.imshow(robot_grid, origin='lower', extent=[min_x, max_x, min_y, max_y],
              cmap=cmap, interpolation='nearest')
    
    # Plot binary occupancy grid
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.imshow(occupancy_grid, origin='lower', extent=[min_x, max_x, min_y, max_y],
              cmap='binary', interpolation='nearest')
    
    # Plot trajectories and points with different colors for each robot
    colors = {'robot1': 'red', 'robot2': 'blue'}
    for robot_id in ['robot1', 'robot2']:
        poses = data[robot_id]['poses']
        x_coords = [pose[0] for pose in poses]
        y_coords = [pose[1] for pose in poses]
        color = colors[robot_id]
        
        # Plot trajectory on both figures
        for ax in [ax1, ax2]:
            ax.plot(x_coords, y_coords, '-', color=color, label=f'{robot_id} trajectory', alpha=0.7)
            ax.plot(x_coords[0], y_coords[0], 'o', color=color, label=f'{robot_id} start')
            ax.plot(x_coords[-1], y_coords[-1], 's', color=color, label=f'{robot_id} end')
    
    # Customize plots
    ax1.set_title('Lidar Readings\n(by Robot)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True)
    ax1.legend()
    ax1.axis('equal')
    
    ax2.set_title('Lidar Readings')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True)
    ax2.legend()
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()
    # plt.savefig('robot_data.pdf')
def main():
    # Load the data
    data = load_robot_data('robot_data.npy')
    
    # Print some basic statistics
    for robot_id in ['robot1', 'robot2']:
        num_poses = len(data[robot_id]['poses'])
        print(f"\n=== {robot_id} Statistics ===")
        print(f"Number of recorded poses: {num_poses}")
        print(f"First pose: {data[robot_id]['poses'][0]}")
        print(f"Last pose: {data[robot_id]['poses'][-1]}")
        print(f"Number of LiDAR readings: {len(data[robot_id]['lidar_readings'])}")
    
    # Plot the data
    plot_robot_trajectories_and_map(data)

if __name__ == "__main__":
    main()
