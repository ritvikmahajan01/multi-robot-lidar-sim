## Merge 2 GPOM predictions using BCM

import numpy as np
from gpom import squash_probability
import matplotlib.pyplot as plt

y_pred1 = np.load("gpom_data/robot1/y_pred_np1.npy")
y_var1 = np.load("gpom_data/robot1/y_var_np1.npy")

y_pred2 = np.load("gpom_data/robot2/y_pred_np2.npy")
y_var2 = np.load("gpom_data/robot2/y_var_np2.npy")


# BCM fusion equations:
# Merged variance = 1 / (1/var1 + 1/var2) for each cell
y_var_merged = 1 / (1/y_var1 + 1/y_var2)

# Merged prediction = var_merged * (pred1/var1 + pred2/var2) for each cell
y_pred_merged = y_var_merged * (y_pred1/y_var1 + y_pred2/y_var2)

# Save the merged results
np.save("gpom_data/y_pred_merged.npy", y_pred_merged)
np.save("gpom_data/y_var_merged.npy", y_var_merged)

# print("BCM fusion completed!")
# print(f"Merged prediction shape: {y_pred_merged.shape}")
# print(f"Merged variance shape: {y_var_merged.shape}")
# print(f"Prediction range: [{y_pred_merged.min():.3f}, {y_pred_merged.max():.3f}]")
# print(f"Variance range: [{y_var_merged.min():.3f}, {y_var_merged.max():.3f}]")




prob_occupied = squash_probability(y_pred_merged, y_var_merged)

occupied_mask = prob_occupied > 0.65
free_mask = prob_occupied < 0.35
unknown_mask = (prob_occupied > 0.35) & (prob_occupied < 0.65)

# Plot Results

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

# Plot the results with raw gp predictions
plt.figure(figsize=(10, 8))
plt.scatter(test_points[:, 0], test_points[:, 1], c=y_pred_merged, cmap='Reds', alpha=0.7, s=0.1, label='GP Predictions')
plt.colorbar(label='Probability')
plt.title('GPOM Predictions')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.legend()
plt.axis('equal')
# plt.show()


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
plt.scatter(test_points[:, 0], test_points[:, 1], c=y_var_merged, cmap='Reds', alpha=0.7, s=0.1, label='Variance')
plt.colorbar(label='Variance')
plt.title('Variance')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.axis('equal')



# Plot classified points
plt.figure(figsize=(10, 8))
plt.scatter(test_points[occupied_mask, 0], test_points[occupied_mask, 1], c='red', alpha=0.7, s=0.5, label='Occupied Points')
plt.scatter(test_points[free_mask, 0], test_points[free_mask, 1], c='blue', alpha=0.7, s=0.5, label='Free Points')
plt.scatter(test_points[unknown_mask, 0], test_points[unknown_mask, 1], c='gray', alpha=0.7, s=0.1, label='Unknown Points')
plt.title('Classified Points')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.axis('equal')
plt.show()
