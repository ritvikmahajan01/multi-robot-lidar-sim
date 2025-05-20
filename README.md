# Lidar Robot Simulation

A modular 2D simulation of two robots with lidar sensing capabilities in an environment with multiple walled obstacles. The code is structured to make it easy to implement custom mapping algorithms.

## Features

- 2D robot simulation with differential drive kinematics
- Two robots with independent lidar sensors
- Lidar sensor simulation with configurable range and resolution
- Interactive environment with multiple walled obstacles
- Real-time visualization of robot positions, orientations, and lidar readings
- Separate keyboard controls for each robot
- Modular code structure for easy extension

## Project Structure

```
lidar_robot_sim/
├── src/
│   ├── robots/
│   │   ├── __init__.py
│   │   └── lidar_robot.py      # Robot class with lidar functionality
│   ├── environment/
│   │   ├── __init__.py
│   │   └── environment.py      # Environment and visualization
│   ├── utils/
│   │   ├── __init__.py
│   │   └── geometry.py         # Geometric utility functions
│   └── main.py                 # Main simulation loop
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.7+
- NumPy
- Pygame

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Simulation

Run the simulation using:
```bash
python src/main.py
```

## Controls

### Red Robot (WASD):
- W: Move forward
- S: Move backward
- A: Rotate counterclockwise
- D: Rotate clockwise

### Blue Robot (Arrow Keys):
- Up Arrow: Move forward
- Down Arrow: Move backward
- Left Arrow: Rotate counterclockwise
- Right Arrow: Rotate clockwise

## Visualization

- Red circle: First robot position
- Blue circle: Second robot position
- Blue line: Robot orientation
- Green lines: Lidar readings
- Black lines: Walls and obstacles

## Extending the Simulation

The code is structured to make it easy to implement custom mapping algorithms:

1. The `LidarRobot` class provides:
   - Robot state (position, orientation)
   - Lidar readings
   - Collision detection

2. The `Environment` class provides:
   - Wall definitions
   - Visualization
   - World-to-screen coordinate conversion

3. The `utils` package provides:
   - Geometric calculations
   - Angle normalization
   - Line intersection detection

To implement a custom mapping algorithm:
1. Create a new class in a new file (e.g., `src/mapping/your_mapping.py`)
2. Use the lidar readings from the robot
3. Update the map based on the robot's pose and sensor data
4. Add visualization of your map in the `Environment` class

## Example: Adding a Simple Mapping Algorithm

```python
# src/mapping/simple_mapping.py
import numpy as np
from typing import List, Tuple

class SimpleMapping:
    def __init__(self, resolution: float = 0.1):
        self.resolution = resolution
        self.map = np.zeros((200, 200))  # 20m x 20m map with 0.1m resolution
        
    def update(self, robot_pose: Tuple[float, float, float], 
               lidar_readings: List[float]) -> None:
        # Update map based on lidar readings
        pass
```

Then in your main simulation:
```python
from mapping.simple_mapping import SimpleMapping

# Initialize mapping
mapping = SimpleMapping()

# In the main loop
for robot in robots:
    mapping.update(robot.get_pose(), robot.lidar_readings)
``` 