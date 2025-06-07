import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

class OccupancyGrid:
    def __init__(self, resolution: float = 0.05):
        """Initialize an empty occupancy grid."""
        self.resolution = resolution
        self.grid = None
        self.bounds = None  # (min_x, max_x, min_y, max_y)
        self.log_odds = None  # For probabilistic updates
        self.robot_detections = None  # Track which robot detected each cell
        
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
        # Initialize robot detections: 0=unknown, 1=robot1, 2=robot2, 3=both
        self.robot_detections = np.zeros((grid_height, grid_width), dtype=np.uint8)
        self.bounds = (min_x, max_x, min_y, max_y)
    
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
                         lidar_readings: List[Tuple[float, float]],
                         robot_id: str) -> None:
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
                self.log_odds[end_grid_y, end_grid_x] += l_occ
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
        """Update cells along a ray using Bresenham's line algorithm."""
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
                self.log_odds[y, x] += l_free
                # Update robot detection for free space
                current_detection = self.robot_detections[y, x]
                if current_detection == 0:
                    self.robot_detections[y, x] = robot_num
                elif current_detection != robot_num:
                    self.robot_detections[y, x] = 3  # Both robots
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def get_occupancy_grid(self) -> np.ndarray:
        """Get the binary occupancy grid."""
        if self.log_odds is None:
            raise ValueError("Grid not initialized")
            
        # Convert log-odds to probability
        prob = 1 - 1 / (1 + np.exp(self.log_odds))
        # Convert to binary grid
        return (prob > 0.5).astype(np.float32)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get the world coordinate bounds of the grid."""
        if self.bounds is None:
            raise ValueError("Grid not initialized")
        return self.bounds

    def visualize_map(self, data: Dict = None, show_probability: bool = False) -> None:
        """Visualize the occupancy grid map with optional robot trajectories."""
        if self.log_odds is None:
            raise ValueError("Grid not initialized")

        # Create figure
        plt.figure(figsize=(12, 10))
        
        if show_probability:
            # Show probability map
            prob = 1 - 1 / (1 + np.exp(self.log_odds))
            plt.imshow(prob, origin='lower', extent=self.bounds,
                      cmap='RdYlBu_r', vmin=0, vmax=1)
            plt.colorbar(label='Occupancy Probability')
        else:
            # Get binary occupancy grid
            occupancy = self.get_occupancy_grid()
            
            # Show black and white occupancy map
            plt.imshow(occupancy, origin='lower', extent=self.bounds,
                      cmap='binary', vmin=0, vmax=1)
            
            # Add legend for occupancy
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='black', label='Occupied Space'),
                Patch(facecolor='white', label='Free Space')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
        
        # Plot robot trajectories if data is provided
        if data is not None:
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
        
        if show_probability:
            plt.title('Occupancy Grid Map by combining data from both robots (Probability)')
        else:
            plt.title('Occupancy Grid Map by combining data from both robots (Binary)')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.grid(True)
        plt.axis('equal')
        plt.show()

def main():
    """Example usage of the OccupancyGrid class."""
    # Load data
    data = np.load('robot_data.npy', allow_pickle=True).item()
    
    # Create and update occupancy grid
    grid = OccupancyGrid(resolution=0.05)
    grid.initialize_grid(data)
    
    # Update grid with all LiDAR readings
    for robot_id in ['robot1', 'robot2']:
        poses = data[robot_id]['poses']
        lidar_readings = data[robot_id]['lidar_readings']
        for pose, readings in zip(poses, lidar_readings):
            grid.update_from_lidar(pose, readings, robot_id)
    
    # Get occupancy grid and robot detections
    occupancy = grid.get_occupancy_grid()
    robot_detections = grid.robot_detections
    
    # Count points for each robot
    for robot_num, robot_id in enumerate(['robot1', 'robot2'], 1):
        # Get mask for cells detected by this robot
        robot_mask = (robot_detections == robot_num) | (robot_detections == 3)
        
        # Count occupied and free points
        occupied_points = np.sum((occupancy == 1) & robot_mask)
        free_points = np.sum((occupancy == 0) & robot_mask)
        
        print(f"\n{robot_id} statistics:")
        print(f"Occupied points: {occupied_points}")
        print(f"Free points: {free_points}")
        print(f"Total points: {occupied_points + free_points}")
    
    # Visualize the map
    grid.visualize_map(data, show_probability=True)
    grid.visualize_map(data, show_probability=False)  # Show robot-specific detections

if __name__ == "__main__":
    main() 