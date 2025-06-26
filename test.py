import numpy as np
import math
import matplotlib.pyplot as plt

import time
from env_utils import ROBOT_CONFIG, ENVIRONMENT_CONFIG, Environment, LidarRobot
from mapping_utils import create_ground_truth_grid, visualize_ground_truth_map

visualize_ground_truth_map(wall_thickness=0.1, resolution=0.05, load_name='ground_truth_mid.npy')