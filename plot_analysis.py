import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define column names
column_names = [
    'timestamp',
    'robot1_time',
    'robot2_time',
    'combined_time',
    'robot1_readings',
    'robot2_readings',
    'total_readings',
    'robot1_total_points',
    'robot2_total_points',
    'total_points'
]

# Read the CSV file with column names
df = pd.read_csv('data/times_combine_ogm.csv', names=column_names)

# Plot 1: Time vs Total Points
plt.figure()
plt.scatter(df['robot1_total_points'], df['robot1_time'], label='Time taken for Robot 1 Occupancy Grid Mapping', alpha=0.6)
plt.scatter(df['robot2_total_points'], df['robot2_time'], label='Time taken for Robot 2 Occupancy Grid Mapping', alpha=0.6)
plt.scatter(df['total_points'], df['combined_time'], label='Time taken for merging the maps', alpha=0.6)
# plt.scatter(df['total_points'], df['combined_time'], label='Time taken for merging the maps', alpha=0.6)
plt.xlabel('Total Points (LiDAR + Free)')
plt.ylabel('Time (seconds)')
plt.title('Processing Time vs Total Points')
plt.legend()
plt.grid(True)

# Plot 2: Time vs LiDAR Readings
plt.figure()
plt.scatter(df['robot1_readings'], df['robot1_time'], label='Robot 1', alpha=0.6)
plt.scatter(df['robot2_readings'], df['robot2_time'], label='Robot 2', alpha=0.6)
plt.scatter(df['total_readings'], df['combined_time'], label='Merging the maps', alpha=0.6)
plt.xlabel('Total LiDAR Readings')
plt.ylabel('Time (seconds)')
plt.title('Processing Time vs LiDAR Readings')
plt.legend()
plt.grid(True)

# Plot 3: Free Points vs LiDAR Points
plt.figure()
robot1_free = df['robot1_total_points'] - df['robot1_readings']
robot2_free = df['robot2_total_points'] - df['robot2_readings']
plt.scatter(df['robot1_readings'], robot1_free, label='Robot 1', alpha=0.6)
plt.scatter(df['robot2_readings'], robot2_free, label='Robot 2', alpha=0.6)
plt.xlabel('LiDAR Readings')
plt.ylabel('Free Points')
plt.title('Free Points vs LiDAR Readings')
plt.legend()
plt.grid(True)

# Plot 4: Ratio of Free Points to LiDAR Readings
plt.figure()
ratio1 = robot1_free / df['robot1_readings']
ratio2 = robot2_free / df['robot2_readings']
plt.scatter(df['robot1_readings'], ratio1, label='Robot 1', alpha=0.6)
plt.scatter(df['robot2_readings'], ratio2, label='Robot 2', alpha=0.6)
plt.xlabel('LiDAR Readings')
plt.ylabel('Ratio (Free Points / LiDAR Readings)')
plt.title('Ratio of Free Points to LiDAR Readings')
plt.legend()
plt.grid(True)
plt.show()

# Print some statistics
print("\nStatistics:")
print(f"Average Robot 1 time: {df['robot1_time'].mean():.3f} seconds")
print(f"Average Robot 2 time: {df['robot2_time'].mean():.3f} seconds")
print(f"Average Combined time: {df['combined_time'].mean():.3f} seconds")
print(f"\nAverage free points per LiDAR reading:")
print(f"Robot 1: {ratio1.mean():.2f}")
print(f"Robot 2: {ratio2.mean():.2f}") 