import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

class RobotOccupancyGrid:
    def __init__(self, resolution: float = 0.05):
        """Initialize an empty occupancy grid for a single robot."""
        self.resolution = resolution
        self.grid = None
        self.bounds = None  # (min_x, max_x, min_y, max_y)
        self.log_odds = None  # For probabilistic updates
        
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
                         lidar_readings: List[Tuple[float, float]]) -> None:
        """Update grid using LiDAR readings."""
        if self.grid is None:
            raise ValueError("Grid not initialized")
            
        robot_x, robot_y, robot_theta = pose
        
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
            
            # Update free cells along the ray
            self._update_ray(robot_x, robot_y, end_x, end_y, l_free)
    
    def _update_ray(self, start_x: float, start_y: float, 
                   end_x: float, end_y: float, l_free: float) -> None:
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
            prob = 1 - 1 / (1 + np.exp(self.log_odds))
            plt.imshow(prob, origin='lower', extent=self.bounds,
                      cmap='RdYlBu_r', vmin=0, vmax=1)
            plt.colorbar(label='Occupancy Probability')
        else:
            # Show binary occupancy grid
            occupancy = self.get_occupancy_grid()
            plt.imshow(occupancy, origin='lower', extent=self.bounds,
                      cmap='binary', vmin=0, vmax=1)
        
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
    
    # Get probability maps
    prob1 = 1 - 1 / (1 + np.exp(grid1.log_odds))
    prob2 = 1 - 1 / (1 + np.exp(grid2.log_odds))
    
    # Initialize combined map
    combined_map = np.zeros_like(prob1)
    
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

def visualize_all_maps(grid1: RobotOccupancyGrid, grid2: RobotOccupancyGrid, data: Dict) -> None:
    """Visualize the combined occupancy grid map."""
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Get combined map
    combined_map = combine_maps(grid1, grid2, data)
    
    # Plot probability map
    im1 = ax1.imshow(combined_map, origin='lower', extent=grid1.bounds,
              cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.colorbar(im1, ax=ax1, label='Occupancy Probability')
    ax1.set_title('Combined Probability Map')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True)
    ax1.axis('equal')
    
    # Plot binary map
    binary_map = (combined_map > 0.5).astype(float)
    im2 = ax2.imshow(binary_map, origin='lower', extent=grid1.bounds,
              cmap='binary', vmin=0, vmax=1)
    plt.colorbar(im2, ax=ax2, label='Occupancy (Binary)')
    ax2.set_title('Combined Binary Map')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True)
    ax2.axis('equal')
    
    # Plot robot trajectories on both maps
    for ax in [ax1, ax2]:
        for robot_id, color in [('robot1', 'red'), ('robot2', 'blue')]:
            poses = data[robot_id]['poses']
            x_coords = [pose[0] for pose in poses]
            y_coords = [pose[1] for pose in poses]
            ax.plot(x_coords, y_coords, '-', label=f'{robot_id} trajectory', 
                    color=color, alpha=0.5)
            ax.plot(x_coords[0], y_coords[0], 'o', label=f'{robot_id} start',
                    color=color)
            ax.plot(x_coords[-1], y_coords[-1], 's', label=f'{robot_id} end',
                    color=color)
        ax.legend()
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # Original visualization code (commented out for later use)
    """
    # Create a figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(30, 20))
    fig.suptitle('Occupancy Grid Maps Comparison', fontsize=16)
    
    # Robot 1 Probability Map
    plt.sca(axes[0, 0])
    grid1.visualize_map(data, 'robot1', show_probability=True)
    
    # Robot 1 Binary Map
    plt.sca(axes[0, 1])
    grid1.visualize_map(data, 'robot1', show_probability=False)
    
    # Robot 2 Probability Map
    plt.sca(axes[1, 0])
    grid2.visualize_map(data, 'robot2', show_probability=True)
    
    # Robot 2 Binary Map
    plt.sca(axes[1, 1])
    grid2.visualize_map(data, 'robot2', show_probability=False)
    
    # Combined Map
    plt.sca(axes[0, 2])
    combined_map = combine_maps(grid1, grid2, data)
    plt.imshow(combined_map, origin='lower', extent=grid1.bounds,
              cmap='RdYlBu_r', vmin=0, vmax=1)
    plt.colorbar(label='Combined Occupancy Probability')
    plt.title('Combined Map')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.axis('equal')
    
    # Plot both robot trajectories on combined map
    for robot_id, color in [('robot1', 'red'), ('robot2', 'blue')]:
        poses = data[robot_id]['poses']
        x_coords = [pose[0] for pose in poses]
        y_coords = [pose[1] for pose in poses]
        plt.plot(x_coords, y_coords, '-', label=f'{robot_id} trajectory', 
                color=color, alpha=0.5)
        plt.plot(x_coords[0], y_coords[0], 'o', label=f'{robot_id} start',
                color=color)
        plt.plot(x_coords[-1], y_coords[-1], 's', label=f'{robot_id} end',
                color=color)
    plt.legend()
    
    # Observation Regions
    plt.sca(axes[1, 2])
    region1 = grid1.get_observation_region('robot1', data)
    region2 = grid2.get_observation_region('robot2', data)
    plt.imshow(region1.astype(int) + region2.astype(int), origin='lower', 
              extent=grid1.bounds, cmap='tab10', vmin=0, vmax=2)
    plt.colorbar(label='Observation Regions (0: None, 1: Robot1, 2: Robot2)')
    plt.title('Observation Regions')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True)
    plt.axis('equal')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    """

def main():
    """Example usage of the RobotOccupancyGrid class."""
    # Load data
    data = np.load('robot_data.npy', allow_pickle=True).item()
    
    # Create separate occupancy grids for each robot
    grid1 = RobotOccupancyGrid(resolution=0.05)
    grid2 = RobotOccupancyGrid(resolution=0.05)
    
    # Initialize grids
    grid1.initialize_grid(data)
    grid2.initialize_grid(data)
    
    # Update grid1 with robot1's LiDAR readings
    poses1 = data['robot1']['poses']
    lidar_readings1 = data['robot1']['lidar_readings']
    for pose, readings in zip(poses1, lidar_readings1):
        grid1.update_from_lidar(pose, readings)
    
    # Update grid2 with robot2's LiDAR readings
    poses2 = data['robot2']['poses']
    lidar_readings2 = data['robot2']['lidar_readings']
    for pose, readings in zip(poses2, lidar_readings2):
        grid2.update_from_lidar(pose, readings)
    
    # Get occupancy grids
    occupancy1 = grid1.get_occupancy_grid()
    occupancy2 = grid2.get_occupancy_grid()
    
    # Count points for each robot
    for grid, robot_id in [(grid1, 'robot1'), (grid2, 'robot2')]:
        occupancy = grid.get_occupancy_grid()
        occupied_points = np.sum(occupancy == 1)
        free_points = np.sum(occupancy == 0)
        
        print(f"\n{robot_id} statistics:")
        print(f"Occupied points: {occupied_points}")
        print(f"Free points: {free_points}")
        print(f"Total points: {occupied_points + free_points}")
    
    # Visualize all maps in parallel
    visualize_all_maps(grid1, grid2, data)

if __name__ == "__main__":
    main()
