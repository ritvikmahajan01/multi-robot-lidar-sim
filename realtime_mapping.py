import pygame
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Tuple, Optional
from env_utils import (
    SIMULATION_CONFIG, ROBOT_CONFIG, ENVIRONMENT_CONFIG,
    LidarRobot, Environment, DataRecorder
)
from combine_ogm_weighted import RobotOccupancyGrid, combine_maps
import time


class RealtimeOccupancyGrid(RobotOccupancyGrid):
    def __init__(self, resolution: float = 0.05):
        super().__init__(resolution)
        # Initialize with fixed bounds based on environment size
        min_x = 0
        max_x =  ENVIRONMENT_CONFIG['x_width'] # Based on environment width
        min_y = 0
        max_y =  ENVIRONMENT_CONFIG['y_height']   # Based on environment height
        
        # Add padding
        # padding = 1.0
        # min_x -= padding
        # min_y -= padding
        # max_x += padding
        # max_y += padding
        
        # Create grid
        grid_width = int((max_x - min_x) / self.resolution)
        grid_height = int((max_y - min_y) / self.resolution)
        self.grid = np.zeros((grid_height, grid_width))
        self.log_odds = np.zeros((grid_height, grid_width))
        self.bounds = (min_x, max_x, min_y, max_y)
        self.robot_detections = np.zeros_like(self.grid, dtype=int)
        self.observed_region = np.zeros_like(self.grid, dtype=bool)  # Track observed cells
        self.free_points_sampled = 0  # Counter for free points sampled
        self.occupied_points_sampled = 0  # Counter for occupied points sampled
    
    def update_from_lidar(self, pose: Tuple[float, float, float], 
                         lidar_readings: List[Tuple[float, float]], robot_id: str) -> None:
        """Update grid using LiDAR readings."""
        if self.grid is None:
            raise ValueError("Grid not initialized")
            
        robot_x, robot_y, robot_theta = pose
        robot_num = 1 if robot_id == 'robot1' else 2
        
        # First update the base class's grid
        super().update_from_lidar(pose, lidar_readings, robot_id)
        
        # Then mark cells as observed
        def mark_cell(x: int, y: int) -> None:
            if 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]:
                self.observed_region[y, x] = True
                self.free_points_sampled += 1
        
        def process_ray(start_x: float, start_y: float, end_x: float, end_y: float, is_max_range: bool = False) -> None:
            # Mark end point if not max range
            if not is_max_range:
                end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)
                if 0 <= end_grid_x < self.grid.shape[1] and 0 <= end_grid_y < self.grid.shape[0]:
                    self.observed_region[end_grid_y, end_grid_x] = True
                    self.occupied_points_sampled += 1
            
            # Sample points along the ray using parent class's method
            self._sample_points_along_ray(start_x, start_y, end_x, end_y, mark_cell)
        
        for range_val, bearing in lidar_readings:
            self._process_lidar_reading(robot_x, robot_y, robot_theta, range_val, bearing, 
                                      robot_num, process_ray)
    
    def get_observed_region(self) -> np.ndarray:
        """Get the region where the robot has gathered data."""
        return self.observed_region

def visualization_process(plot_queue, stop_event):
    """Process for handling visualization updates"""
    plt.ion()
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    
    # Set titles
    ax1.set_title('Robot 1 Map')
    ax2.set_title('Robot 2 Map')
    ax3.set_title('Combined Map')
    
    # Initialize plot objects
    map1_plot = ax1.imshow(np.zeros((100, 100)), cmap='RdYlBu_r', vmin=0, vmax=1, origin='lower')
    map2_plot = ax2.imshow(np.zeros((100, 100)), cmap='RdYlBu_r', vmin=0, vmax=1, origin='lower')
    map3_plot = ax3.imshow(np.zeros((100, 100)), cmap='RdYlBu_r', vmin=0, vmax=1, origin='lower')
    
    robot1_pos, = ax1.plot([], [], 'ro', markersize=10)
    robot2_pos, = ax2.plot([], [], 'bo', markersize=10)
    robot1_pos_combined, = ax3.plot([], [], 'ro', markersize=10)
    robot2_pos_combined, = ax3.plot([], [], 'bo', markersize=10)
    
    # Set up the plots
    for ax in [ax1, ax2, ax3]:
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
    
    # Position the window
    fig.canvas.manager.window.setGeometry(100, 100, 1500, 500)
    
    while not stop_event.is_set():
        try:
            # Non-blocking check for new data
            if not plot_queue.empty():
                data = plot_queue.get_nowait()
                grid1, grid2, combined_grid, r1_x, r1_y, r2_x, r2_y = data
                
                # Update map plots
                map1_plot.set_data(grid1)
                map2_plot.set_data(grid2)
                map3_plot.set_data(combined_grid)
                
                # Update robot positions
                robot1_pos.set_data([r1_x], [r1_y])
                robot2_pos.set_data([r2_x], [r2_y])
                robot1_pos_combined.set_data([r1_x], [r2_y])
                robot2_pos_combined.set_data([r2_x], [r2_y])
                
                # Update the plot
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
            
            # Small sleep to prevent CPU overuse
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Visualization error: {e}")
            continue

class RealtimeMappingSimulation:
    def __init__(self):
        """Initialize the simulation"""
        pygame.init()
        self.screen = pygame.display.set_mode((ENVIRONMENT_CONFIG['width'], ENVIRONMENT_CONFIG['height']))
        pygame.display.set_caption("Robot Simulation")
        
        # Initialize environment
        self.env = Environment(
            width=ENVIRONMENT_CONFIG['width'],
            height=ENVIRONMENT_CONFIG['height'],
            screen=self.screen
        )
        
        # Initialize robots with positions for 12x8 map
        self.robot1 = LidarRobot(
            x=40.0,  # Start near left side
            y=200.0,  # Start in middle vertically
            theta=0,
            color=ENVIRONMENT_CONFIG['colors']['robot1'],
            **ROBOT_CONFIG
        )
        
        self.robot2 = LidarRobot(
            x=560.0,  # Start near right side
            y=200.0,   # Start in middle vertically
            theta=np.pi,
            color=ENVIRONMENT_CONFIG['colors']['robot2'],
            **ROBOT_CONFIG
        )
        
        # Create list of robots for drawing
        self.robots = [self.robot1, self.robot2]
        
        # Initialize occupancy grids
        self.grid1 = RealtimeOccupancyGrid()
        self.grid2 = RealtimeOccupancyGrid()
        
        # Initialize data recorder
        self.data_recorder = DataRecorder('robot_data.npy')
        
        # Initialize matplotlib figure for maps
        plt.ion()  # Enable interactive mode
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 6))  # Increased figure size
        self.fig.canvas.manager.set_window_title('Occupancy Grid Maps')
        
        # Calculate grid dimensions for padded 14x10 meters with 0.05m resolution
        grid_width = int(14 / 0.05)   # 280 cells (12m + 2m padding)
        grid_height = int(10 / 0.05)  # 200 cells (8m + 2m padding)
        
        # Set up the map plots with correct extents for padded map
        self.map1_plot = self.ax1.imshow(
            np.zeros((grid_height, grid_width)),
            cmap='RdYlBu_r',
            vmin=0,
            vmax=1,
            extent=[-1, 13, -1, 9],  # [xmin, xmax, ymin, ymax] in meters with padding
            origin='lower'
        )
        self.map2_plot = self.ax2.imshow(
            np.zeros((grid_height, grid_width)),
            cmap='RdYlBu_r',
            vmin=0,
            vmax=1,
            extent=[-1, 13, -1, 9],
            origin='lower'
        )
        self.map3_plot = self.ax3.imshow(
            np.zeros((grid_height, grid_width)),
            cmap='RdYlBu_r',
            vmin=0,
            vmax=1,
            extent=[-1, 13, -1, 9],
            origin='lower'
        )
        
        # Plot robot positions
        self.robot1_pos, = self.ax1.plot([], [], 'ro', markersize=5)
        self.robot2_pos, = self.ax2.plot([], [], 'bo', markersize=5)
        self.robot1_pos_combined, = self.ax3.plot([], [], 'ro', markersize=5)
        self.robot2_pos_combined, = self.ax3.plot([], [], 'bo', markersize=5)
        
        # Set titles and labels
        self.ax1.set_title('Robot 1 Map')
        self.ax2.set_title('Robot 2 Map')
        self.ax3.set_title('Combined Map')
        
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.grid(True)
            ax.set_aspect('equal')
            ax.set_xlim(-1, 13)  # Set fixed x limits with padding
            ax.set_ylim(-1, 9)   # Set fixed y limits with padding
        
        # Adjust layout to maximize map display area
        plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.2)
        plt.show(block=False)
        
        # Initialize data dictionary for map combination
        self.data = {
            'robot1': {'poses': [], 'lidar_readings': [], 'angles': []},
            'robot2': {'poses': [], 'lidar_readings': [], 'angles': []}
        }
        
        # Set up the clock for controlling frame rate
        self.clock = pygame.time.Clock()
        
        # Initialize last update time
        self.last_update_time = time.time()
    
    def update_maps(self):
        """Update the occupancy grid maps"""
        # Update grid1 with robot1's data
        if self.robot1.lidar_readings:
            angles = np.linspace(-ROBOT_CONFIG['lidar_angle_range']/2,
                               ROBOT_CONFIG['lidar_angle_range']/2,
                               len(self.robot1.lidar_readings))
            self.grid1.update_from_lidar(
                self.robot1.get_true_pose(),
                list(zip(self.robot1.lidar_readings, angles)),
                'robot1'
            )
        
        # Update grid2 with robot2's data
        if self.robot2.lidar_readings:
            angles = np.linspace(-ROBOT_CONFIG['lidar_angle_range']/2,
                               ROBOT_CONFIG['lidar_angle_range']/2,
                               len(self.robot2.lidar_readings))
            self.grid2.update_from_lidar(
                self.robot2.get_true_pose(),
                list(zip(self.robot2.lidar_readings, angles)),
                'robot2'
            )
        
        # Get current occupancy grids
        grid1 = self.grid1.get_occupancy_grid()
        grid2 = self.grid2.get_occupancy_grid()
        
        # Combine maps by adding log-odds
        combined_log_odds = self.grid1.log_odds + self.grid2.log_odds
        combined_prob = 1 / (1 + np.exp(-combined_log_odds))
        
        # Convert to ternary grid
        combined_grid = np.zeros_like(combined_prob)
        combined_grid[combined_prob >= 0.9] = 1.0  # Occupied
        combined_grid[combined_prob <= 0.1] = 0.0  # Free
        combined_grid[(combined_prob >= 0.1) & (combined_prob <= 0.9)] = 0.5  # Unknown
        
        # Update map plots
        self.map1_plot.set_data(grid1)
        self.map2_plot.set_data(grid2)
        self.map3_plot.set_data(combined_grid)
        
        # Update robot positions
        r1_x, r1_y = self.robot1.get_true_pose()[:2]
        r2_x, r2_y = self.robot2.get_true_pose()[:2]
        
        # Update robot positions in their respective maps
        self.robot1_pos.set_data([r1_x], [r1_y])
        self.robot2_pos.set_data([r2_x], [r2_y])
        
        # Update robot positions in combined map
        self.robot1_pos_combined.set_data([r1_x], [r1_y])
        self.robot2_pos_combined.set_data([r2_x], [r2_y])
        
        # Update the plot
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def run(self):
        """Run the simulation"""
        running = True
        while running:
            # Calculate delta time
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_s:
                        self.data_recorder.save_data()
            
            # Handle keyboard input for robot 1 (WASD)
            keys = pygame.key.get_pressed()
            self.robot1.velocity = 0
            self.robot1.angular_velocity = 0
            
            if keys[pygame.K_w]:
                self.robot1.velocity = ROBOT_CONFIG['max_velocity']
            if keys[pygame.K_s]:
                self.robot1.velocity = -ROBOT_CONFIG['max_velocity']
            if keys[pygame.K_a]:
                self.robot1.angular_velocity = ROBOT_CONFIG['max_angular_velocity']
            if keys[pygame.K_d]:
                self.robot1.angular_velocity = -ROBOT_CONFIG['max_angular_velocity']
            
            # Handle keyboard input for robot 2 (Arrow keys)
            self.robot2.velocity = 0
            self.robot2.angular_velocity = 0
            
            if keys[pygame.K_UP]:
                self.robot2.velocity = ROBOT_CONFIG['max_velocity']
            if keys[pygame.K_DOWN]:
                self.robot2.velocity = -ROBOT_CONFIG['max_velocity']
            if keys[pygame.K_LEFT]:
                self.robot2.angular_velocity = ROBOT_CONFIG['max_angular_velocity']
            if keys[pygame.K_RIGHT]:
                self.robot2.angular_velocity = -ROBOT_CONFIG['max_angular_velocity']
            
            # Update robot states
            for robot in self.robots:
                robot.update(dt, self.env.get_walls())
                robot.get_lidar_readings(self.env.get_walls())
            
            # Update maps
            self.update_maps()
            
            # Draw everything
            self.env.draw(self.robots)
            
            # Update display
            pygame.display.flip()
            
            # Control frame rate
            self.clock.tick(SIMULATION_CONFIG['fps'])
        
        # Save recorded data
        self.data_recorder.save_data()
        pygame.quit()

if __name__ == "__main__":
    simulation = RealtimeMappingSimulation()
    simulation.run() 