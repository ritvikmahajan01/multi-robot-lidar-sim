# Multi-Robot LiDAR Simulation

A Python-based simulation environment for multiple robots equipped with LiDAR sensors. This project demonstrates collaborative estimation and navigation in a shared environment.

## Features

- Multiple robots with LiDAR sensors
- Real-time visualization of robot positions and LiDAR readings
- Collision detection and avoidance
- Uncertainty visualization
- Interactive control using keyboard inputs

## Requirements

- Python 3.7 or higher
- Pygame
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ritvikmahajan01/multi-robot-lidar-sim.git
cd multi-robot-lidar-sim
```

2. Install the required packages:
```bash
pip install pygame numpy
```

## Usage

Run the simulation:
```bash
python main.py
```

### Controls

- **Robot 1 (Red)**:
  - W: Move forward
  - S: Move backward
  - A: Rotate left
  - D: Rotate right

- **Robot 2 (Blue)**:
  - Arrow Up: Move forward
  - Arrow Down: Move backward
  - Arrow Left: Rotate left
  - Arrow Right: Rotate right

- **General**:
  - Space: Toggle wall visibility
  - ESC: Quit simulation

## Project Structure

- `main.py`: Main simulation code containing robot and environment classes
- `definitions.py`: Configuration parameters and constants
- `.gitignore`: Git ignore rules for Python projects

<!-- ## Contributing

Feel free to submit issues and enhancement requests! -->

<!-- ## License

This project is open source and available under the MIT License.  -->