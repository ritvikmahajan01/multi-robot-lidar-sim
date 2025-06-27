import numpy as np
from mapping_utils import RobotOccupancyGrid, create_ground_truth_grid, evaluate_map, combine_grid_maps
import matplotlib.pyplot as plt

def main():
    # Load data
    data = np.load('robot_data_large.npy', allow_pickle=True).item()
    
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
    ground_truth = np.load('ground_truth_large.npy', allow_pickle=True)

    # Plot ground truth
    # plt.imshow(ground_truth)
    # plt.show()
    
    # Get observation regions
    region1 = grid1.get_observation_region('robot1', data)
    region2 = grid2.get_observation_region('robot2', data)
    
    # Combine maps
    combined_map = combine_grid_maps(grid1, grid2, data)
    
    # Evaluate each map
    metrics1 = evaluate_map(grid1.get_occupancy_grid(), ground_truth, region1)
    metrics2 = evaluate_map(grid2.get_occupancy_grid(), ground_truth, region2)
    
    # For combined map, evaluate regions that are properly combined
    # This matches the logic in combine_maps:
    # - Regions seen only by robot 1
    # - Regions seen only by robot 2
    # - Regions seen by both robots

    combined_region = region1 | region2
    metrics_combined = evaluate_map(combined_map, ground_truth, combined_region)
    
    # Print results
    print("\n=== Robot 1 Map Evaluation ===")
    print("ROC AUC: ", metrics1['roc_auc'], "NLL: ", metrics1['nll'], "Unknown Percentage: ", metrics1['unknown_percentage'])
    # for metric, value in metrics1.items():
    #     print(f"{metric}: {value}")
    
    print("\n=== Robot 2 Map Evaluation ===")
    print("ROC AUC: ", metrics2['roc_auc'], "NLL: ", metrics2['nll'], "Unknown Percentage: ", metrics2['unknown_percentage'])
    # for metric, value in metrics2.items():
    #     print(f"{metric}: {value}")
    
    print("\n=== Combined Map Evaluation ===")
    print("ROC AUC: ", metrics_combined['roc_auc'], "NLL: ", metrics_combined['nll'], "Unknown Percentage: ", metrics_combined['unknown_percentage'])
    # for metric, value in metrics_combined.items():
    #     print(f"{metric}: {value}")

if __name__ == "__main__":
    main() 