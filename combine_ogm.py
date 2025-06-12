import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
import os
import csv
from datetime import datetime

class RobotOccupancyGrid:
    def __init__(self, resolution: float = 0.05):
        """Initialize an empty occupancy grid for a single robot."""
        self.resolution = resolution
        self.grid = None
        self.bounds = None  # (min_x, max_x, min_y, max_y)
        self.log_odds = None  # For probabilistic updates
        self.occupied_threshold = 0.9
        self.free_threshold = 0.1
        self.robot_detections = None  # Track which robot detected each cell
        self.free_points_sampled = 0  # Counter for free points sampled
        
    def initialize_grid(self, data: Dict) -> None:
        """Initialize grid dimensions based on robot trajectories."""
        # Find map boundaries
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for robot_id in ['robot1', 'robot2']:
            poses = data[robot_id]['poses']
            for pose in poses:
                min_x = min(min_x, pose[0])
                max_x = max(max_x, pose[0])
                min_y = min(min_y, pose[1])
                max_y = max(max_y, pose[1])
        
        # Add padding
        padding = 1.0
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        
        # Create grid
        grid_width = int((max_x - min_x) / self.resolution)
        grid_height = int((max_y - min_y) / self.resolution)
        self.grid = np.zeros((grid_height, grid_width))
        self.log_odds = np.zeros((grid_height, grid_width))
        self.bounds = (min_x, max_x, min_y, max_y)
        self.robot_detections = np.zeros_like(self.grid, dtype=int)  # Initialize robot_detections
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        if self.bounds is None:
            raise ValueError("Grid not initialized")
            
        min_x, max_x, min_y, max_y = self.bounds
        grid_x = int((x - min_x) / self.resolution)
        grid_y = int((y - min_y) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        if self.bounds is None:
            raise ValueError("Grid not initialized")
            
        min_x, max_x, min_y, max_y = self.bounds
        x = min_x + (grid_x + 0.5) * self.resolution
        y = min_y + (grid_y + 0.5) * self.resolution
        return x, y
    
    def update_from_lidar(self, pose: Tuple[float, float, float], 
                         lidar_readings: List[Tuple[float, float]], robot_id: str) -> None:
        """Update grid using LiDAR readings."""
        if self.grid is None:
            raise ValueError("Grid not initialized")
            
        robot_x, robot_y, robot_theta = pose
        robot_num = 1 if robot_id == 'robot1' else 2
        
        # Constants for log-odds update
        l_occ = 0.7  # Log-odds for occupied cells
        l_free = -0.4  # Log-odds for free cells
        
        for range_val, bearing in lidar_readings:
            # Skip readings at max range (no obstacle detected)
            if range_val >= 3:
                continue
                
            # Calculate end point of LiDAR ray
            bearing_rad = math.radians(bearing)
            end_x = robot_x + range_val * math.cos(robot_theta + bearing_rad)
            end_y = robot_y + range_val * math.sin(robot_theta + bearing_rad)
            
            # Get grid coordinates
            end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
            
            # Update occupied cell
            if 0 <= end_grid_x < self.grid.shape[1] and 0 <= end_grid_y < self.grid.shape[0]:
                # Update log-odds with bounds to prevent overflow
                self.log_odds[end_grid_y, end_grid_x] = np.clip(
                    self.log_odds[end_grid_y, end_grid_x] + l_occ,
                    -100, 100  # Clip log-odds to prevent overflow
                )
                # Update robot detection
                current_detection = self.robot_detections[end_grid_y, end_grid_x]
                if current_detection == 0:
                    self.robot_detections[end_grid_y, end_grid_x] = robot_num
                elif current_detection != robot_num:
                    self.robot_detections[end_grid_y, end_grid_x] = 3  # Both robots
            
            # Update free cells along the ray
            self._update_ray(robot_x, robot_y, end_x, end_y, l_free, robot_num)
    
    def _update_ray(self, start_x: float, start_y: float, 
                   end_x: float, end_y: float, l_free: float, robot_num: int) -> None:
        """Update cells along a ray from start to end point."""
        # Get grid coordinates
        start_grid_x, start_grid_y = self.world_to_grid(start_x, start_y)
        end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
        
        # Use Bresenham's line algorithm to get cells along the ray
        dx = abs(end_grid_x - start_grid_x)
        dy = abs(end_grid_y - start_grid_y)
        sx = 1 if start_grid_x < end_grid_x else -1
        sy = 1 if start_grid_y < end_grid_y else -1
        err = dx - dy
        
        x, y = start_grid_x, start_grid_y
        while x != end_grid_x or y != end_grid_y:
            if 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]:
                # Update log-odds with bounds to prevent overflow
                self.log_odds[y, x] = np.clip(
                    self.log_odds[y, x] + l_free,
                    -100, 100  # Clip log-odds to prevent overflow
                )
                # Update robot detection
                current_detection = self.robot_detections[y, x]
                if current_detection == 0:
                    self.robot_detections[y, x] = robot_num
                elif current_detection != robot_num:
                    self.robot_detections[y, x] = 3  # Both robots
                self.free_points_sampled += 1  # Increment counter for each free point
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def get_occupancy_grid(self) -> np.ndarray:
        """Get the ternary occupancy grid (0=free, 1=occupied, 0.5=unknown) using sigmoid function."""
        if self.log_odds is None:
            raise ValueError("Grid not initialized")
            
        # Convert log odds to probabilities using sigmoid function
        probabilities = 1 / (1 + np.exp(-self.log_odds))
        
        # Create ternary grid based on probability thresholds
        grid = np.zeros_like(self.log_odds)
        grid[probabilities > self.occupied_threshold] = 1.0  # Occupied
        grid[probabilities < self.free_threshold] = 0.0  # Free
        grid[(probabilities >= self.free_threshold) & (probabilities <= self.occupied_threshold)] = 0.5  # Unknown
        return grid
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get the world coordinate bounds of the grid."""
        if self.bounds is None:
            raise ValueError("Grid not initialized")
        return self.bounds

    def get_observation_region(self, robot_id: str, data: Dict) -> np.ndarray:
        """Get the region where the robot has gathered data."""
        if self.grid is None:
            raise ValueError("Grid not initialized")
            
        # Initialize observation region mask
        observation_region = np.zeros_like(self.grid, dtype=bool)
        
        # Get robot trajectory
        poses = data[robot_id]['poses']
        lidar_readings = data[robot_id]['lidar_readings']
        
        # For each pose and its corresponding lidar readings
        for pose, readings in zip(poses, lidar_readings):
            robot_x, robot_y, robot_theta = pose
            
            # For each lidar reading
            for range_val, bearing in readings:
                if range_val >= 3:  # Skip max range readings
                    continue
                    
                # Calculate end point of LiDAR ray
                bearing_rad = math.radians(bearing)
                end_x = robot_x + range_val * math.cos(robot_theta + bearing_rad)
                end_y = robot_y + range_val * math.sin(robot_theta + bearing_rad)
                
                # Get grid coordinates
                end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
                
                # Mark the region as observed
                if 0 <= end_grid_x < self.grid.shape[1] and 0 <= end_grid_y < self.grid.shape[0]:
                    observation_region[end_grid_y, end_grid_x] = True
                    
                # Mark cells along the ray as observed
                self._mark_ray_region(robot_x, robot_y, end_x, end_y, observation_region)
        
        return observation_region
    
    def _mark_ray_region(self, start_x: float, start_y: float, 
                        end_x: float, end_y: float, region_mask: np.ndarray) -> None:
        """Mark cells along a ray as observed using Bresenham's line algorithm."""
        start_grid_x, start_grid_y = self.world_to_grid(start_x, start_y)
        end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
        
        # Bresenham's line algorithm
        dx = abs(end_grid_x - start_grid_x)
        dy = abs(end_grid_y - start_grid_y)
        sx = 1 if start_grid_x < end_grid_x else -1
        sy = 1 if start_grid_y < end_grid_y else -1
        err = dx - dy
        
        x, y = start_grid_x, start_grid_y
        while x != end_grid_x or y != end_grid_y:
            if 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]:
                region_mask[y, x] = True
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def visualize_map(self, data: Dict = None, robot_id: str = None, show_probability: bool = False) -> None:
        """Visualize the occupancy grid map with optional robot trajectory."""
        if self.log_odds is None:
            raise ValueError("Grid not initialized")

        if show_probability:
            # Show probability map
            prob = 1 / (1 + np.exp(-self.log_odds))
            plt.imshow(prob, origin='lower', extent=self.bounds,
                      cmap='RdYlBu_r', vmin=0, vmax=1)
            plt.colorbar(label='Occupancy Probability')
        else:
            # Get ternary occupancy grid
            occupancy = self.get_occupancy_grid()
            
            # Show ternary occupancy map with custom colormap
            cmap = plt.cm.get_cmap('RdYlBu_r', 3)
            plt.imshow(occupancy, origin='lower', extent=self.bounds,
                      cmap=cmap, vmin=0, vmax=1)
            
            # Add legend for occupancy
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label='Occupied Space'),
                Patch(facecolor='yellow', label='Unknown Space'),
                Patch(facecolor='blue', label='Free Space')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
        
        # Plot robot trajectory if data is provided
        if data is not None and robot_id is not None:
            poses = data[robot_id]['poses']
            x_coords = [pose[0] for pose in poses]
            y_coords = [pose[1] for pose in poses]
            color = 'red' if robot_id == 'robot1' else 'blue'
            plt.plot(x_coords, y_coords, '-', label=f'{robot_id} trajectory', 
                    color=color, alpha=0.5)
            plt.plot(x_coords[0], y_coords[0], 'o', label=f'{robot_id} start',
                    color=color)
            plt.plot(x_coords[-1], y_coords[-1], 's', label=f'{robot_id} end',
                    color=color)
        
        plt.title(f'Occupancy Grid Map - {robot_id}')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()

def combine_maps(grid1: RobotOccupancyGrid, grid2: RobotOccupancyGrid, data: Dict) -> np.ndarray:
    """Combine maps from two robots using weighted probabilities."""
    # Get observation regions for each robot
    region1 = grid1.get_observation_region('robot1', data)
    region2 = grid2.get_observation_region('robot2', data)
    
    # Get probability maps using sigmoid function
    prob1 = 1 / (1 + np.exp(-grid1.log_odds))
    prob2 = 1 / (1 + np.exp(-grid2.log_odds))
    
    # Initialize combined map
    combined_map = np.ones_like(prob1) * 0.5
    
    # Region where only robot1 has observations
    only_robot1 = region1 & ~region2
    combined_map[only_robot1] = prob1[only_robot1]
    
    # Region where only robot2 has observations
    only_robot2 = ~region1 & region2
    combined_map[only_robot2] = prob2[only_robot2]
    
    # Region where both robots have observations
    both_robots = region1 & region2
    
    # Calculate weights based on number of observations
    # This is a simple weighting - could be improved with more sophisticated metrics
    weight1 = np.sum(region1) / (np.sum(region1) + np.sum(region2))
    weight2 = 1 - weight1
    
    # Combine probabilities in overlapping regions
    combined_map[both_robots] = weight1 * prob1[both_robots] + weight2 * prob2[both_robots]
    
    return combined_map

def visualize_all_maps(grid1: RobotOccupancyGrid, grid2: RobotOccupancyGrid, combined_map: np.ndarray, data: Dict) -> None:
    """Visualize all maps separately."""
    # 1. Robot 1's probability map
    plt.figure(figsize=(12, 10))
    grid1.visualize_map(data, 'robot1', show_probability=True)
    plt.title('Robot 1 - Occupancy Grid Map (Probability)')
    plt.show()

    # 2. Robot 1's binary map
    plt.figure(figsize=(12, 10))
    grid1.visualize_map(data, 'robot1', show_probability=False)
    plt.title('Robot 1 - Occupancy Grid Map (Binary)')
    plt.show()

    # 3. Robot 2's probability map
    plt.figure(figsize=(12, 10))
    grid2.visualize_map(data, 'robot2', show_probability=True)
    plt.title('Robot 2 - Occupancy Grid Map (Probability)')
    plt.show()

    # 4. Robot 2's binary map
    plt.figure(figsize=(12, 10))
    grid2.visualize_map(data, 'robot2', show_probability=False)
    plt.title('Robot 2 - Occupancy Grid Map (Binary)')
    plt.show()

    # 5. Combined probability map
    plt.figure(figsize=(12, 10))
    plt.imshow(combined_map, origin='lower', extent=grid1.bounds,
              cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.colorbar(label='Occupancy Probability')
    
    # Plot both robot trajectories
    for robot_id in ['robot1', 'robot2']:
        poses = data[robot_id]['poses']
        x_coords = [pose[0] for pose in poses]
        y_coords = [pose[1] for pose in poses]
        color = 'red' if robot_id == 'robot1' else 'blue'
        plt.plot(x_coords, y_coords, '-', label=f'{robot_id} trajectory', 
                color=color, alpha=0.5)
        plt.plot(x_coords[0], y_coords[0], 'o', label=f'{robot_id} start',
                color=color)
        plt.plot(x_coords[-1], y_coords[-1], 's', label=f'{robot_id} end',
                color=color)
    
    plt.title('Combined Occupancy Grid Map (Probability) using weighted average of individual probability maps')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

    # 6. Combined binary map
    plt.figure(figsize=(12, 10))
    # Create ternary grid based on probability thresholds
    ternary_map = np.zeros_like(combined_map)
    ternary_map[combined_map > 0.65] = 1.0  # Occupied
    ternary_map[combined_map < 0.35] = 0.0  # Free
    ternary_map[(combined_map >= 0.35) & (combined_map <= 0.65)] = 0.5  # Unknown
    
    # Show ternary occupancy map with custom colormap
    cmap = plt.cm.get_cmap('RdYlBu_r', 3)
    plt.imshow(ternary_map, origin='lower', extent=grid1.bounds,
              cmap=cmap, vmin=0, vmax=1)
    
    # Add legend for occupancy
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Occupied Space'),
        Patch(facecolor='yellow', label='Unknown Space'),
        Patch(facecolor='blue', label='Free Space')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Plot both robot trajectories
    for robot_id in ['robot1', 'robot2']:
        poses = data[robot_id]['poses']
        x_coords = [pose[0] for pose in poses]
        y_coords = [pose[1] for pose in poses]
        color = 'red' if robot_id == 'robot1' else 'blue'
        plt.plot(x_coords, y_coords, '-', label=f'{robot_id} trajectory', 
                color=color, alpha=0.5)
        plt.plot(x_coords[0], y_coords[0], 'o', label=f'{robot_id} start',
                color=color)
        plt.plot(x_coords[-1], y_coords[-1], 's', label=f'{robot_id} end',
                color=color)
    
    plt.title('Combined Occupancy Grid Map (Ternary) using weighted average of individual probability maps')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

    # Show observation regions in different figure
    # plt.figure(figsize=(12, 10))
    # region1 = grid1.get_observation_region('robot1', data)
    # region2 = grid2.get_observation_region('robot2', data)
    # plt.imshow(region1.astype(int) + region2.astype(int), origin='lower', 
    #           extent=grid1.bounds, cmap='tab10', vmin=0, vmax=2)
    # plt.colorbar(label='Observation Regions (0: None, 1: Robot1, 2: Robot2)')
    # plt.title('Observation Regions')
    # plt.show()

def main():
    """Example usage of the RobotOccupancyGrid class."""
    # Load data
    data = np.load('robot_data.npy', allow_pickle=True).item()
    
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
    combined_map = combine_maps(grid1, grid2, data)
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
    # visualize_all_maps(grid1, grid2, combined_map, data)

if __name__ == "__main__":
    main()