import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
import time
from env_utils import ROBOT_CONFIG, ENVIRONMENT_CONFIG, Environment
# ROC and NLL
from sklearn.metrics import roc_curve, roc_auc_score, log_loss
from skimage.draw import line

default_resolution = 0.05

class RobotOccupancyGrid:
    def __init__(self, resolution: float = default_resolution):
        """Initialize an empty occupancy grid for a single robot."""
        self.resolution = resolution
        self.grid = None
        self.bounds = None  # (min_x, max_x, min_y, max_y)
        self.log_odds = None  # For probabilistic updates
        
        # Constants for occupancy grid
        self.occupied_threshold = 0.9
        self.free_threshold = 0.1
        self.max_range = ROBOT_CONFIG['lidar_range']  # Maximum range for LiDAR readings
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
        min_x = 0
        max_x = ENVIRONMENT_CONFIG['x_width']
        min_y = 0
        max_y = ENVIRONMENT_CONFIG['y_height']
        
        
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
    
    # def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
    #     """Convert grid coordinates to world coordinates."""
    #     if self.bounds is None:
    #         raise ValueError("Grid not initialized")
            
    #     min_x, max_x, min_y, max_y = self.bounds
    #     x = min_x + (grid_x + 0.5) * self.resolution
    #     y = min_y + (grid_y + 0.5) * self.resolution
    #     return x, y

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






def combine_grid_maps(grid1: RobotOccupancyGrid, grid2: RobotOccupancyGrid, data: Dict) -> np.ndarray:
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



### Evaluation

def create_ground_truth_grid(wall_thickness: float = None, resolution: float = 0.05, save = False, save_name: str = 'ground_truth.npy') -> np.ndarray:
    """Create a ground truth occupancy grid from the walls defined in Environment.
    
    Args:
        wall_thickness: Physical thickness of walls in meters. If None, uses ENVIRONMENT_CONFIG['wall_thickness']
        resolution: Grid resolution in meters. If None, uses grid.resolution
    """
    if wall_thickness is None:
        wall_thickness = ENVIRONMENT_CONFIG['wall_thickness']

    env = Environment(width=ENVIRONMENT_CONFIG['width'], height=ENVIRONMENT_CONFIG['height'])
    walls = env.get_walls()

    x_width = ENVIRONMENT_CONFIG['x_width']
    y_height = ENVIRONMENT_CONFIG['y_height']

    grid_x_width = x_width / resolution
    grid_y_height = y_height / resolution

    grid_x_width = int(grid_x_width)
    grid_y_height = int(grid_y_height)

    ground_truth = np.zeros((grid_y_height, grid_x_width))
    
    # Calculate wall thickness in grid cells
    wall_thickness_cells = max(1, int(wall_thickness / resolution))

    def convert_to_grid_index(x: float, y: float) -> Tuple[int, int]:
        return int(x / resolution), int(y / resolution)
    
    for wall in walls:
        start_x, start_y = wall[0]
        end_x, end_y = wall[1]
        
        # Convert to grid coordinates
        start_grid_x, start_grid_y = convert_to_grid_index(start_x, start_y)
        end_grid_x, end_grid_y = convert_to_grid_index(end_x, end_y)
        
        # Calculate wall direction vector
        wall_dx = end_grid_x - start_grid_x
        wall_dy = end_grid_y - start_grid_y
        wall_length = np.sqrt(wall_dx**2 + wall_dy**2)
        
        if wall_length == 0:
            continue  # Skip zero-length walls
        
        # Normalize direction vector
        wall_dx /= wall_length
        wall_dy /= wall_length
        
        # Calculate perpendicular vector for thickness
        perp_dx = -wall_dy
        perp_dy = wall_dx
        
        # Calculate half-thickness offset
        half_thickness = wall_thickness_cells / 2
        
        # Calculate the four corners of the thick wall rectangle
        corners = []
        
        # Top-left corner (start + perpendicular offset)
        corners.append((
            int(start_grid_x + perp_dx * half_thickness),
            int(start_grid_y + perp_dy * half_thickness)
        ))
        
        # Top-right corner (start - perpendicular offset)
        corners.append((
            int(start_grid_x - perp_dx * half_thickness),
            int(start_grid_y - perp_dy * half_thickness)
        ))
        
        # Bottom-right corner (end - perpendicular offset)
        corners.append((
            int(end_grid_x - perp_dx * half_thickness),
            int(end_grid_y - perp_dy * half_thickness)
        ))
        
        # Bottom-left corner (end + perpendicular offset)
        corners.append((
            int(end_grid_x + perp_dx * half_thickness),
            int(end_grid_y + perp_dy * half_thickness)
        ))
        
        # Convert corners to polygon format for skimage.draw.polygon
        # skimage.draw.polygon expects (row, col) format
        polygon_rows = [corner[1] for corner in corners]
        polygon_cols = [corner[0] for corner in corners]
        
        # Draw the thick wall using polygon
        from skimage.draw import polygon
        rr, cc = polygon(polygon_rows, polygon_cols, shape=ground_truth.shape)
        
        # Mark all cells in the polygon as occupied
        for y, x in zip(rr, cc):
            if 0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]:
                ground_truth[y, x] = 1.0

    # Save the ground truth map
    if save:
        np.save(save_name, ground_truth)
    
    return ground_truth

def visualize_ground_truth_map(wall_thickness: float = None, resolution: float = 0.05, load_name: str = None) -> None:
    """Visualize the ground truth occupancy grid map.
    
    Args:
        wall_thickness: Physical thickness of walls in meters. If None, uses ENVIRONMENT_CONFIG['wall_thickness']
        resolution: Grid resolution in meters. If None, uses grid.resolution
        title: Title for the plot
    """
    if wall_thickness is None:
        wall_thickness = ENVIRONMENT_CONFIG['wall_thickness']

    # Create ground truth map
    if load_name is None:
        ground_truth = create_ground_truth_grid(wall_thickness, resolution, save=False)
    else:
        ground_truth = np.load(load_name)

    x_width = ENVIRONMENT_CONFIG['x_width']
    y_height = ENVIRONMENT_CONFIG['y_height']
    
    # Create the plot
    plt.figure()
    
    # Display the ground truth map with proper extent
    plt.imshow(ground_truth, origin='lower',
              cmap='binary', extent=[0, x_width, 0, y_height])
    
    # Setup plot
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.show()



def evaluate_map(predicted_map: np.ndarray, ground_truth: np.ndarray, 
                observation_region: np.ndarray) -> Dict[str, float]:
    """Evaluate a map against ground truth, considering only observed regions.
    
    Args:
        predicted_map: Either a ternary occupancy grid (0=free, 1=occupied, 0.5=unknown)
                      or a probability map (0.0 to 1.0)
        ground_truth: Binary ground truth map (0=free, 1=occupied)
        observation_region: Boolean mask of observed regions
    """
    # Consider only observed regions
    mask = observation_region.astype(bool)
    
    # Calculate metrics
    total_observed = np.sum(mask)
    if total_observed == 0:
        return {
            'accuracy': 0.0,
            'false_negatives': 0.0,
            'false_positives': 0.0,
            'unknown_percentage': 0.0,
            'observed_area_percentage': 0.0
        }
    
    # # Convert probability map to ternary if needed
    # if np.any((predicted_map > 0) & (predicted_map < 1) & (predicted_map != 0.5)):
    #     # This is a probability map, convert to ternary
    ternary_map = np.zeros_like(predicted_map)
    ternary_map[predicted_map > 0.9] = 1.0  # Occupied
    ternary_map[predicted_map < 0.1] = 0.0  # Free
    ternary_map[(predicted_map >= 0.1) & (predicted_map <= 0.9)] = 0.5  # Unknown


    # predicted_map = ternary_map
    
    # Create mask for cells that are not unknown
    known_mask = (ternary_map != 0.5) & mask
    
    # True positives: correctly identified occupied cells
    tp = np.sum((ternary_map == 1) & (ground_truth == 1) & known_mask)
    
    # True negatives: correctly identified free cells
    tn = np.sum((ternary_map == 0) & (ground_truth == 0) & known_mask)
    
    # False positives: incorrectly identified as occupied
    fp = np.sum((ternary_map == 1) & (ground_truth == 0) & known_mask)
    
    # False negatives: incorrectly identified as free
    fn = np.sum((ternary_map == 0) & (ground_truth == 1) & known_mask)
    
    # Unknown cells in observed region
    unknown = np.sum((ternary_map == 0.5) & mask)
    
    # Calculate total valid cells (excluding unknown)
    total_valid = tp + tn + fp + fn
    
    # Calculate metrics only if there are valid cells to evaluate
    if total_valid > 0:
        accuracy = (tp + tn) / total_valid
        false_negatives = fn / total_valid
        false_positives = fp / total_valid
    else:
        accuracy = 0.0
        false_negatives = 0.0
        false_positives = 0.0
    
    unknown_percentage = unknown / total_observed
    observed_area_percentage = total_observed / (ground_truth.shape[0] * ground_truth.shape[1])


    known_ground_truth = ground_truth[known_mask]   
    known_predicted_map = predicted_map[known_mask]
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(known_ground_truth, known_predicted_map)
    roc_auc = roc_auc_score(known_ground_truth, known_predicted_map)


    nll = log_loss(known_ground_truth, known_predicted_map)
    
    metrics = {
        'accuracy': accuracy,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'unknown_percentage': unknown_percentage,
        'observed_area_percentage': observed_area_percentage,
        'roc_auc': roc_auc,
        'nll': nll,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }

    # Plot ROC curve
    # plt.figure(figsize=(10, 8))
    # plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve')
    # plt.legend()
    # plt.show()
    
    return metrics