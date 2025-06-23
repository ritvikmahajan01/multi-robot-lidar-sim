import numpy as np
from typing import Dict, List, Tuple
from utils import Environment
from combine_ogm_weighted import RobotOccupancyGrid, combine_maps

def create_ground_truth_grid(grid: RobotOccupancyGrid) -> np.ndarray:
    """Create a ground truth occupancy grid from the walls defined in Environment."""
    # Create environment to get walls
    env = Environment(width=1200, height=800)
    walls = env.get_walls()
    
    # Create a new grid with the same dimensions as the input grid
    ground_truth = np.zeros_like(grid.grid)
    
    # Convert walls to grid coordinates and mark as occupied
    for wall in walls:
        start_x, start_y = wall[0]
        end_x, end_y = wall[1]
        
        # Convert to grid coordinates
        start_grid_x, start_grid_y = grid.world_to_grid(start_x, start_y)
        end_grid_x, end_grid_y = grid.world_to_grid(end_x, end_y)
        
        # Mark cells along the wall as occupied
        def mark_cell(x: int, y: int) -> None:
            if 0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]:
                ground_truth[y, x] = 1.0
        
        # Mark start point as occupied if within bounds
        if 0 <= start_grid_x < ground_truth.shape[1] and 0 <= start_grid_y < ground_truth.shape[0]:
            ground_truth[start_grid_y, start_grid_x] = 1.0
        
        # Mark end point as occupied if within bounds
        if 0 <= end_grid_x < ground_truth.shape[1] and 0 <= end_grid_y < ground_truth.shape[0]:
            ground_truth[end_grid_y, end_grid_x] = 1.0
        
        # Sample points along the wall and mark them as occupied
        grid._sample_points_along_ray(start_x, start_y, end_x, end_y, mark_cell)
    
    return ground_truth

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
    predicted_map = ternary_map
    
    # Create mask for cells that are not unknown
    known_mask = (predicted_map != 0.5) & mask
    
    # True positives: correctly identified occupied cells
    tp = np.sum((predicted_map == 1) & (ground_truth == 1) & known_mask)
    
    # True negatives: correctly identified free cells
    tn = np.sum((predicted_map == 0) & (ground_truth == 0) & known_mask)
    
    # False positives: incorrectly identified as occupied
    fp = np.sum((predicted_map == 1) & (ground_truth == 0) & known_mask)
    
    # False negatives: incorrectly identified as free
    fn = np.sum((predicted_map == 0) & (ground_truth == 1) & known_mask)
    
    # Unknown cells in observed region
    unknown = np.sum((predicted_map == 0.5) & mask)
    
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
    
    metrics = {
        'accuracy': accuracy,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'unknown_percentage': unknown_percentage,
        'observed_area_percentage': observed_area_percentage
    }
    
    return metrics

def main():
    # Load data
    data = np.load('robot_data.npy', allow_pickle=True).item()
    
    # Create and update occupancy grid for robot 1
    grid1 = RobotOccupancyGrid(resolution=0.05)
    grid1.initialize_grid(data)
    poses1 = data['robot1']['poses']
    lidar_readings1 = data['robot1']['lidar_readings']
    for pose, readings in zip(poses1, lidar_readings1):
        grid1.update_from_lidar(pose, readings, 'robot1')
    
    # Create and update occupancy grid for robot 2
    grid2 = RobotOccupancyGrid(resolution=0.05)
    grid2.initialize_grid(data)
    poses2 = data['robot2']['poses']
    lidar_readings2 = data['robot2']['lidar_readings']
    for pose, readings in zip(poses2, lidar_readings2):
        grid2.update_from_lidar(pose, readings, 'robot2')
    
    # Create ground truth grid using grid1's dimensions
    ground_truth = create_ground_truth_grid(grid1)
    
    # Get observation regions
    region1 = grid1.get_observation_region('robot1', data)
    region2 = grid2.get_observation_region('robot2', data)
    
    # Combine maps
    combined_map = combine_maps(grid1, grid2, data)
    
    # Evaluate each map
    metrics1 = evaluate_map(grid1.get_occupancy_grid(), ground_truth, region1)
    metrics2 = evaluate_map(grid2.get_occupancy_grid(), ground_truth, region2)
    
    # For combined map, evaluate regions that are properly combined
    # This matches the logic in combine_maps:
    # - Regions seen only by robot 1
    # - Regions seen only by robot 2
    # - Regions seen by both robots
    only_region1 = region1 & ~region2
    only_region2 = region2 & ~region1
    overlap = region1 & region2
    combined_region = only_region1 | only_region2 | overlap
    metrics_combined = evaluate_map(combined_map, ground_truth, combined_region)
    
    # Print results
    print("\n=== Robot 1 Map Evaluation ===")
    for metric, value in metrics1.items():
        print(f"{metric}: {value:.2%}")
    
    print("\n=== Robot 2 Map Evaluation ===")
    for metric, value in metrics2.items():
        print(f"{metric}: {value:.2%}")
    
    print("\n=== Combined Map Evaluation ===")
    for metric, value in metrics_combined.items():
        print(f"{metric}: {value:.2%}")

if __name__ == "__main__":
    main() 