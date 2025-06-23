import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
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
        
        # Constants for occupancy grid
        self.occupied_threshold = 0.9
        self.free_threshold = 0.1
        self.max_range = 3.0  # Maximum range for LiDAR readings
        self.points_per_ray = 20  # Fixed number of points to sample per ray
        
        # Constants for log-odds update
        self.l_occ = 0.7  # Log-odds for occupied cells
        self.l_free = -0.4  # Log-odds for free cells
        
        # Tracking variables
        self.robot_detections = None  # Track which robot detected each cell
        self.free_points_sampled = 0  # Counter for free points sampled
        self.occupied_points_sampled = 0  # Counter for occupied points sampled
    
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
        self.robot_detections = np.zeros_like(self.grid, dtype=int)
    
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

    def _process_lidar_reading(self, robot_x: float, robot_y: float, robot_theta: float,
                             range_val: float, bearing: float, robot_num: int,
                             callback: Callable[[float, float, float, float], None]) -> None:
        """Process a single LiDAR reading.
        
        Args:
            robot_x, robot_y, robot_theta: Robot pose
            range_val, bearing: LiDAR reading
            robot_num: Robot identifier (1 or 2)
            callback: Function to call with start and end points of the ray
        """
        # Calculate end point of LiDAR ray
        bearing_rad = math.radians(bearing)
        end_x = robot_x + range_val * math.cos(robot_theta + bearing_rad)
        end_y = robot_y + range_val * math.sin(robot_theta + bearing_rad)
        
        # For max range readings, use max_range as the end point but don't mark it as occupied
        if range_val >= self.max_range:
            end_x = robot_x + self.max_range * math.cos(robot_theta + bearing_rad)
            end_y = robot_y + self.max_range * math.sin(robot_theta + bearing_rad)
            # Pass a flag to indicate this is a max range reading
            callback(robot_x, robot_y, end_x, end_y, is_max_range=True)
        else:
            callback(robot_x, robot_y, end_x, end_y, is_max_range=False)
    
    def _sample_points_along_ray(self, start_x: float, start_y: float, end_x: float, end_y: float,
                               callback: Callable[[int, int], None]) -> None:
        """Sample fixed number of points along a ray and call callback for each point.
        
        Args:
            start_x, start_y: Start point in world coordinates
            end_x, end_y: End point in world coordinates
            callback: Function to call for each sampled point
        """
        for i in range(self.points_per_ray):
            t = (i + 1) / (self.points_per_ray + 1)  # Parameter from 0 to 1
            sample_x = start_x + t * (end_x - start_x)
            sample_y = start_y + t * (end_y - start_y)
            
            # Convert to grid coordinates and call callback
            grid_x, grid_y = self.world_to_grid(sample_x, sample_y)
            if 0 <= grid_x < self.grid.shape[1] and 0 <= grid_y < self.grid.shape[0]:
                callback(grid_x, grid_y)
    
    def _is_max_range(self, start_x: float, start_y: float, end_x: float, end_y: float) -> bool:
        """Check if the ray length exceeds maximum range."""
        return np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) >= self.max_range
    
    def _update_robot_detection(self, grid_x: int, grid_y: int, robot_num: int) -> None:
        """Update which robot detected a cell."""
        current_detection = self.robot_detections[grid_y, grid_x]
        if current_detection == 0:
            self.robot_detections[grid_y, grid_x] = robot_num
        elif current_detection != robot_num:
            self.robot_detections[grid_y, grid_x] = 3  # Both robots
    
    def update_from_lidar(self, pose: Tuple[float, float, float], 
                         lidar_readings: List[Tuple[float, float]], robot_id: str) -> None:
        """Update grid using LiDAR readings."""
        if self.grid is None:
            raise ValueError("Grid not initialized")
            
        robot_x, robot_y, robot_theta = pose
        robot_num = 1 if robot_id == 'robot1' else 2
        
        def update_cell(x: int, y: int) -> None:
            # Update log-odds with bounds to prevent overflow
            self.log_odds[y, x] = np.clip(
                self.log_odds[y, x] + self.l_free,
                -100, 100  # Clip log-odds to prevent overflow
            )
            self._update_robot_detection(x, y, robot_num)
            self.free_points_sampled += 1
        
        def process_ray(start_x: float, start_y: float, end_x: float, end_y: float, is_max_range: bool = False) -> None:
            # Update occupied cell only if it's not a max range reading
            if not is_max_range:
                end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
                if 0 <= end_grid_x < self.grid.shape[1] and 0 <= end_grid_y < self.grid.shape[0]:
                    self.log_odds[end_grid_y, end_grid_x] = np.clip(
                        self.log_odds[end_grid_y, end_grid_x] + self.l_occ,
                        -100, 100
                    )
                    self._update_robot_detection(end_grid_x, end_grid_y, robot_num)
                    self.occupied_points_sampled += 1
            
            # Sample points along the ray
            self._sample_points_along_ray(start_x, start_y, end_x, end_y, update_cell)
        
        for range_val, bearing in lidar_readings:
            self._process_lidar_reading(robot_x, robot_y, robot_theta, range_val, bearing, 
                                      robot_num, process_ray)
    
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
        
        def mark_cell(x: int, y: int) -> None:
            observation_region[y, x] = True
        
        def process_ray(start_x: float, start_y: float, end_x: float, end_y: float, is_max_range: bool = False) -> None:
            # Mark end point if it's not a max range reading
            if not is_max_range:
                end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
                if 0 <= end_grid_x < self.grid.shape[1] and 0 <= end_grid_y < self.grid.shape[0]:
                    observation_region[end_grid_y, end_grid_x] = True
            
            # Sample points along the ray
            self._sample_points_along_ray(start_x, start_y, end_x, end_y, mark_cell)
        
        # For each pose and its corresponding lidar readings
        for pose, readings in zip(data[robot_id]['poses'], data[robot_id]['lidar_readings']):
            robot_x, robot_y, robot_theta = pose
            for range_val, bearing in readings:
                self._process_lidar_reading(robot_x, robot_y, robot_theta, range_val, bearing, 
                                          1, process_ray)
        
        return observation_region

    def _plot_robot_trajectory(self, data: Dict, robot_id: str, color: str) -> None:
        """Plot robot trajectory with start and end points."""
        poses = data[robot_id]['poses']
        x_coords = [pose[0] for pose in poses]
        y_coords = [pose[1] for pose in poses]
        plt.plot(x_coords, y_coords, '-', label=f'{robot_id} trajectory', 
                color=color, alpha=0.5)
        plt.plot(x_coords[0], y_coords[0], 'o', label=f'{robot_id} start',
                color=color)
        plt.plot(x_coords[-1], y_coords[-1], 's', label=f'{robot_id} end',
                color=color)

    def _setup_plot(self, title: str) -> None:
        """Setup common plot parameters."""
        plt.title(title)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()

    def visualize_map(self, data: Dict = None, robot_id: str = None, show_probability: bool = False) -> None:
        """Visualize the occupancy grid map with optional robot trajectory."""
        if self.log_odds is None:
            raise ValueError("Grid not initialized")

        # Remove redundant figure creation since it's already created in visualize_all_maps
        # plt.figure(figsize=(12, 10))
        
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
            color = 'red' if robot_id == 'robot1' else 'blue'
            self._plot_robot_trajectory(data, robot_id, color)
        
        self._setup_plot(f'Occupancy Grid Map - {robot_id}')

def combine_maps(grid1: RobotOccupancyGrid, grid2: RobotOccupancyGrid, data: Dict) -> np.ndarray:
    """Combine maps from two robots by adding log-odds."""
    # Get observation regions
    region1 = grid1.get_observation_region('robot1', data)
    region2 = grid2.get_observation_region('robot2', data)
    
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

def visualize_all_maps(grid1: RobotOccupancyGrid, grid2: RobotOccupancyGrid, combined_map: np.ndarray, data: Dict) -> None:
    """Visualize all maps separately."""
    # 1. Robot 1's probability map
    plt.figure(figsize=(12, 10))
    grid1.visualize_map(data, 'robot1', show_probability=True)
    # plt.show()

    # 2. Robot 1's binary map
    plt.figure(figsize=(12, 10))
    grid1.visualize_map(data, 'robot1', show_probability=False)
    # plt.show()

    # 3. Robot 2's probability map
    plt.figure(figsize=(12, 10))
    grid2.visualize_map(data, 'robot2', show_probability=True)
    # plt.show()

    # 4. Robot 2's binary map
    plt.figure(figsize=(12, 10))
    grid2.visualize_map(data, 'robot2', show_probability=False)
    # plt.show()

    # 5. Combined probability map
    plt.figure(figsize=(12, 10))
    plt.imshow(combined_map, origin='lower', extent=grid1.bounds,
              cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.colorbar(label='Occupancy Probability')
    
    # Plot both robot trajectories
    grid1._plot_robot_trajectory(data, 'robot1', 'red')
    grid1._plot_robot_trajectory(data, 'robot2', 'blue')
    
    grid1._setup_plot('Combined Occupancy Grid Map (Probability) using weighted average of individual probability maps')
    # plt.show()

    # 6. Combined binary map
    plt.figure(figsize=(12, 10))
    # Create ternary grid based on probability thresholds
    ternary_map = np.zeros_like(combined_map)
    ternary_map[combined_map > 0.9] = 1.0  # Occupied
    ternary_map[combined_map < 0.1] = 0.0  # Free
    ternary_map[(combined_map >= 0.1) & (combined_map <= 0.9)] = 0.5  # Unknown
    
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
    grid1._plot_robot_trajectory(data, 'robot1', 'red')
    grid1._plot_robot_trajectory(data, 'robot2', 'blue')
    
    grid1._setup_plot('Combined Occupancy Grid Map (Ternary) using weighted average of individual probability maps')
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
    
    # # Prepare data for CSV
    # csv_data = {
    #     'timestamp': timestamp,
    #     'robot1_time': end_time1 - start_time1,
    #     'robot2_time': end_time2 - start_time2,
    #     'combined_time': end_time_combined - start_time_combined,
    #     'robot1_readings': robot1_readings,
    #     'robot2_readings': robot2_readings,
    #     'total_readings': total_readings,
    #     'robot1_total_points': robot1_total,
    #     'robot2_total_points': robot2_total,
    #     'total_points': total_points
    # }
    
    # # Write to CSV file
    # csv_file = 'data/times_combine_ogm.csv'
    # file_exists = os.path.isfile(csv_file)
    
    # with open(csv_file, mode='a', newline='') as f:
    #     writer = csv.DictWriter(f, fieldnames=csv_data.keys())
    #     if not file_exists:
    #         writer.writeheader()
    #     writer.writerow(csv_data)
    
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