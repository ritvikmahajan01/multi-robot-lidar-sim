import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import math
import torch as t
import gpytorch as gp
from scipy.stats import norm
import time



def create_point_dataset(data: Dict, free_per_beam: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a dataset of free and occupied points from robot data.
    
    Args:
        data: Dictionary containing robot poses and LiDAR readings
        
    Returns:
        Tuple of (free_points, occupied_points) where:
        - free_points: Nx2 array of (x,y) coordinates of free points
        - occupied_points: Mx2 array of (x,y) coordinates of occupied points
    """

    all_free_points = []
    all_occupied_points = []

    poses = data['poses']
    lidar_readings = data['lidar_readings']
    
    for pose, readings in zip(poses, lidar_readings):
        robot_x, robot_y, robot_theta = pose
        free_points = []

        for range_val, bearing in readings:
            bearing_rad = math.radians(bearing)
            
            # For max range readings, use max_range as the end point
            if range_val >= 3.0:
                end_x = robot_x + 3.0 * math.cos(robot_theta + bearing_rad)
                end_y = robot_y + 3.0 * math.sin(robot_theta + bearing_rad)
            else:
                end_x = robot_x + range_val * math.cos(robot_theta + bearing_rad)
                end_y = robot_y + range_val * math.sin(robot_theta + bearing_rad)
                all_occupied_points.append((end_x, end_y))

            # Sample free points along the ray
            for i in range(free_per_beam):
                t = (i + 1) / (free_per_beam + 1)  # Parameter from 0 to 1
                x = robot_x + t * (end_x - robot_x)
                y = robot_y + t * (end_y - robot_y)
                free_points.append((x, y))

        all_free_points.extend(free_points)
    
    return (np.array(all_free_points), 
            np.array(all_occupied_points))



def visualize_point_dataset(free_points: np.ndarray, 
                          occupied_points: np.ndarray,) -> None:
    """
    Visualize the free and occupied points dataset along with robot trajectories.
    
    Args:
        free_points: Nx2 array of (x,y) coordinates of free points
        occupied_points: Mx2 array of (x,y) coordinates of occupied points
    """
    plt.figure(figsize=(10, 8))
    
    plt.scatter(free_points[:,0], free_points[:,1],
               c='pink', alpha=0.3, s=0.5, label='Free Points')


    
    plt.scatter(occupied_points[:,0], occupied_points[:,1],
               c='red', alpha=0.7, s=0.1, label='Occupied Points')

    
    plt.title('Free and Occupied Points Dataset')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    # plt.grid(True)
    plt.axis('equal')
    # plt.show()
    

def squash_probability(mean: float, variance: float, alpha: float = 100.0, beta: float = 0.0) -> float:
    """
    Improved probability squashing function that takes into account sensor uncertainty.
    
    Args:
        mean: GP mean prediction
        variance: GP variance prediction
        alpha: Scaling factor for mean
        beta: Bias term
    """
    # Add sensor noise to variance
    sensor_noise = 0.0  # Estimated sensor noise
    total_variance = variance + sensor_noise
    
    # Compute probability with improved numerical stability
    numerator = alpha * mean + beta
    denominator = np.sqrt(1 + (alpha ** 2) * total_variance)
    return norm.cdf(numerator / denominator)

class ExactGPModel(gp.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        
        # Use Matern kernel with lengthscale prior
        self.covar_module = gp.kernels.MaternKernel(nu=1.5)


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp_model(model, likelihood, train_x, train_y, device):
    """Train GP model with improved training process."""
    model.train()
    likelihood.train()
    
    # Use Adam optimizer instead of LBFGS for better stability
    optimizer = t.optim.Adam(model.parameters(), lr=0.01)
    mll = gp.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Training parameters
    n_epochs = 100
    best_loss = float('inf')
    patience = 10
    no_improve_count = 0
    
    def closure():
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        return loss
    
    for epoch in range(n_epochs):
        # Optimize
        loss = optimizer.step(closure)
        
        # Early stopping with validation
        if loss.item() < best_loss:
            best_loss = loss.item()
            no_improve_count = 0
            # Save best model
            best_state = {
                'model': model.state_dict(),
                'likelihood': likelihood.state_dict(),
                'loss': best_loss
            }
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            # Restore best model
            model.load_state_dict(best_state['model'])
            likelihood.load_state_dict(best_state['likelihood'])
            break
            
        # if epoch % 10 == 0:
        print(f'Epoch {epoch+1}/{n_epochs} - Loss: {loss.item():.3f}')
        print(f'Lengthscale: {model.covar_module.lengthscale.item():.3f}')
        print(f'Noise: {model.likelihood.noise.item():.3f}')

def main():
    train = 0
    store_results = 0

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the data
    data = np.load('robot_data.npy', allow_pickle=True).item()
    
    # Create point dataset
    free_points, occupied_points = create_point_dataset(data['robot2'], free_per_beam=5)

    print(free_points.shape)
    print(occupied_points.shape)

    # Reduce number of points to prevent memory issues
    temp_num_of_points = 10000  # Reduced from 10000

    if train:
        try:
            # Clear CUDA cache before training
            if device.type == 'cuda':
                t.cuda.empty_cache()

            # Randomly select points
            free_indices = np.random.choice(len(free_points), temp_num_of_points, replace=False)
            occupied_indices = np.random.choice(len(occupied_points), temp_num_of_points, replace=False)
        
            # Convert data to tensors and move to device
            train_x = t.tensor(np.concatenate([free_points[free_indices], occupied_points[occupied_indices]]), dtype=t.float32).to(device)
            train_y = t.tensor(np.concatenate([np.ones(temp_num_of_points) * -1, np.ones(temp_num_of_points)]), dtype=t.float32).to(device)

            start_time = time.time()
            # Initialize likelihood and model
            likelihood = gp.likelihoods.GaussianLikelihood().to(device)
            model = ExactGPModel(train_x, train_y, likelihood).to(device)

            # Train model with improved training process
            train_gp_model(model, likelihood, train_x, train_y, device)

            end_time = time.time()
            print(f"Time taken to train the model: {end_time - start_time} seconds")

            # Save model
            t.save(model.state_dict(), 'model2_state.pth')
            t.save(likelihood.state_dict(), 'likelihood2.pth')    
            np.save('train_x2.npy', train_x.cpu().numpy())
            np.save('train_y2.npy', train_y.cpu().numpy())

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA out of memory error. Try reducing the number of points or using CPU.")
                if device.type == 'cuda':
                    t.cuda.empty_cache()
                # Fallback to CPU
                device = t.device('cpu')
                print(f"Switching to device: {device}")
                # Retry with CPU
                train_x = train_x.cpu()
                train_y = train_y.cpu()
                model = model.cpu()
                likelihood = likelihood.cpu()
                train_gp_model(model, likelihood, train_x, train_y, device)
            else:
                raise e

    elif store_results:
        train_x = t.tensor(np.load('train_x2.npy'), dtype=t.float32).to(device)
        train_y = t.tensor(np.load('train_y2.npy'), dtype=t.float32).to(device)

        likelihood = gp.likelihoods.GaussianLikelihood().to(device)
        model = ExactGPModel(train_x, train_y, likelihood).to(device)

        # Load model
        model.load_state_dict(t.load('model2_state.pth'))
        likelihood.load_state_dict(t.load('likelihood2.pth'))

     # Map size

        start_time = time.time()
        x_min = 0
        x_max = 12
        y_min = 0
        y_max = 8

        # Test points on a grid in the range of the map
        resolution = 0.05
        x = np.arange(x_min, x_max, resolution)
        y = np.arange(y_min, y_max, resolution)
        xx, yy = np.meshgrid(x, y)
        test_points = np.column_stack((xx.flatten(), yy.flatten()))
        test_x = t.tensor(test_points, dtype=t.float32).to(device)

        model.eval()
        likelihood.eval()
        with t.no_grad(), gp.settings.fast_pred_var():
            f_pred = model(test_x)
            y_pred = likelihood(f_pred)

        # Convert predictions to numpy arrays
        # f_pred_np = f_pred.mean.cpu().numpy()
        y_pred_np = y_pred.mean.cpu().numpy()
        y_var_np = y_pred.variance.cpu().numpy()
        end_time = time.time()
        print(f"Time taken to predict: {end_time - start_time} seconds")

        ## Save predictions
        np.save('y_pred_np2.npy', y_pred_np)
        np.save('y_var_np2.npy', y_var_np)

    else:
        y_pred_np = np.load('y_pred_np1.npy')
        y_var_np = np.load('y_var_np1.npy')

        train_x = np.load('train_x1.npy')
        train_y = np.load('train_y1.npy')
        # Map size
        x_min = 0
        x_max = 12
        y_min = 0
        y_max = 8

        # Test points on a grid in the range of the map
        resolution = 0.05
        x = np.arange(x_min, x_max, resolution)
        y = np.arange(y_min, y_max, resolution)
        xx, yy = np.meshgrid(x, y)
        test_points = np.column_stack((xx.flatten(), yy.flatten()))
        test_x = t.tensor(test_points, dtype=t.float32).to(device)


        # Plot the results with raw gp predictions
        plt.figure(figsize=(10, 8))
        plt.scatter(test_points[:, 0], test_points[:, 1], c=y_pred_np, cmap='Reds', alpha=0.7, s=0.1, label='GP Predictions')
        plt.colorbar(label='Probability')
        plt.title('GPOM Predictions')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()
        plt.axis('equal')
        # plt.show()

        # Probability of being occupied
        prob_occupied = squash_probability(y_pred_np, y_var_np)
        # prob_free = 1 - prob_occupied

        # Plot probability of being occupied
        plt.figure(figsize=(10, 8))
        plt.scatter(test_points[:, 0], test_points[:, 1], c=prob_occupied, cmap='Reds', alpha=0.7, s=0.1, label='Probability of being occupied')
        plt.colorbar(label='Probability')
        plt.title('Probability of being occupied')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
        # plt.show()

        # Plot variance
        plt.figure(figsize=(10, 8))
        plt.scatter(test_points[:, 0], test_points[:, 1], c=y_var_np, cmap='Reds', alpha=0.7, s=0.1, label='Variance')
        plt.colorbar(label='Variance')
        plt.title('Variance')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')

        # Classify points
        occupied_mask = prob_occupied > 0.65
        free_mask = prob_occupied < 0.35
        unknown_mask = (prob_occupied > 0.35) & (prob_occupied < 0.65)
        
        # Plot classified points
        plt.figure(figsize=(10, 8))
        plt.scatter(test_points[occupied_mask, 0], test_points[occupied_mask, 1], c='red', alpha=0.7, s=0.5, label='Occupied Points')
        plt.scatter(test_points[free_mask, 0], test_points[free_mask, 1], c='blue', alpha=0.7, s=0.5, label='Free Points')
        plt.scatter(test_points[unknown_mask, 0], test_points[unknown_mask, 1], c='gray', alpha=0.7, s=0.1, label='Unknown Points')
        plt.title('Classified Points')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.axis('equal')
        # plt.show()




        visualize_point_dataset(train_x[train_y == -1], train_x[train_y == 1])
        plt.show()


if __name__ == "__main__":
    main()