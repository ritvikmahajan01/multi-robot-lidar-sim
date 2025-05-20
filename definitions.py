import pygame
import math
import numpy as np
from typing import List, Tuple, Optional, Dict

# Configuration
SIMULATION_CONFIG = {
    'fps': 60,
    'dt': 1/60,
}

ROBOT_CONFIG = {
    'max_velocity': 2.0,
    'max_angular_velocity': 2.0,
    'lidar_range': 3.0,
    'lidar_angle_range': 180,
    'lidar_resolution': 1,
    'radius': 0.1,  # Increased robot size for better visibility
    'sensor_noise': {
        'range_std': 0.0,  # Disabled range noise
        'angle_std': 0.0,  # Disabled angle noise
        'dropout_prob': 0.0,  # Disabled dropouts
        'min_range': 0.1,
        'max_range': 5.0
    },
    'motion_noise': {
        'velocity_std': 0.0,  # Disabled velocity noise
        'angular_std': 0.0,  # Disabled angular noise
        'slip_prob': 0.0  # Disabled wheel slip
    }
}

ENVIRONMENT_CONFIG = {
    'width': 1200,  # Reduced width
    'height': 800,  # Reduced height
    'colors': {
        'robot1': (255, 0, 0),    # Red
        'robot2': (0, 0, 255),    # Blue
        'robot3': (0, 255, 0),    # Green
        'robot4': (255, 165, 0)   # Orange
    }
}

class LidarRobot:
    def __init__(self, x: float, y: float, theta: float, color: Tuple[int, int, int], **config):
        self.x = x
        self.y = y
        self.theta = theta
        self.velocity = 0.0
        self.angular_velocity = 0.0
        self.lidar_range = config.get('lidar_range', 5.0)
        self.lidar_angle_range = config.get('lidar_angle_range', 180)
        self.lidar_resolution = config.get('lidar_resolution', 1)
        self.lidar_readings = []
        self.lidar_uncertainties = []
        self.color = color
        self.radius = config.get('radius', 0.3)
        
        # Noise parameters
        self.sensor_noise = config.get('sensor_noise', {})
        self.motion_noise = config.get('motion_noise', {})
        
        # Ground truth pose
        self.true_x = x
        self.true_y = y
        self.true_theta = theta

    def update(self, dt: float, walls: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> None:
        """Update robot state with collision detection and motion noise."""
        # Add motion noise
        noisy_velocity = self.velocity + np.random.normal(0, self.motion_noise.get('velocity_std', 0))
        noisy_angular_velocity = self.angular_velocity + np.random.normal(0, self.motion_noise.get('angular_std', 0))
        
        # Simulate wheel slip
        if np.random.random() < self.motion_noise.get('slip_prob', 0):
            noisy_velocity *= 0.5
            noisy_angular_velocity *= 0.5
        
        # Calculate new position
        new_x = self.true_x + noisy_velocity * math.cos(self.true_theta) * dt
        new_y = self.true_y + noisy_velocity * math.sin(self.true_theta) * dt
        new_theta = self.true_theta + noisy_angular_velocity * dt
        new_theta = math.atan2(math.sin(new_theta), math.cos(new_theta))

        # Check for collisions with walls
        if not self._check_collision(new_x, new_y, walls):
            self.true_x = new_x
            self.true_y = new_y
            self.true_theta = new_theta
            
            # Update noisy pose estimate
            self.x = self.true_x + np.random.normal(0, self.motion_noise.get('velocity_std', 0))
            self.y = self.true_y + np.random.normal(0, self.motion_noise.get('velocity_std', 0))
            self.theta = self.true_theta + np.random.normal(0, self.motion_noise.get('angular_std', 0))
            self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

    def get_lidar_readings(self, walls: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> Tuple[List[float], List[float]]:
        """Get lidar readings with noise and uncertainty."""
        readings = []
        uncertainties = []
        angles = np.linspace(-self.lidar_angle_range/2, self.lidar_angle_range/2, 
                           int(self.lidar_angle_range/self.lidar_resolution))
        
        for angle in angles:
            # Add angle noise
            noisy_angle = angle + np.random.normal(0, self.sensor_noise.get('angle_std', 0))
            ray_angle = math.radians(noisy_angle) + self.true_theta
            
            # Calculate ray end point
            ray_end_x = self.true_x + self.lidar_range * math.cos(ray_angle)
            ray_end_y = self.true_y + self.lidar_range * math.sin(ray_angle)
            
            # Find closest intersection with walls
            min_distance = self.lidar_range
            for wall in walls:
                intersection = self._line_intersection(
                    (self.true_x, self.true_y), (ray_end_x, ray_end_y),
                    wall[0], wall[1]
                )
                if intersection:
                    distance = math.sqrt((intersection[0] - self.true_x)**2 + 
                                       (intersection[1] - self.true_y)**2)
                    min_distance = min(min_distance, distance)
            
            # Add range noise and handle dropouts
            if np.random.random() < self.sensor_noise.get('dropout_prob', 0):
                readings.append(self.lidar_range)
                uncertainties.append(float('inf'))
            else:
                noisy_distance = min_distance + np.random.normal(0, self.sensor_noise.get('range_std', 0))
                noisy_distance = max(self.sensor_noise.get('min_range', 0.1),
                                   min(self.sensor_noise.get('max_range', 5.0), noisy_distance))
                readings.append(noisy_distance)
                uncertainty = (self.sensor_noise.get('range_std', 0) * 
                             (1 + min_distance / self.lidar_range))
                uncertainties.append(uncertainty)
        
        self.lidar_readings = readings
        self.lidar_uncertainties = uncertainties
        return readings, uncertainties

    def get_pose(self) -> Tuple[float, float, float]:
        """Get current robot pose (x, y, theta)."""
        return (self.x, self.y, self.theta)

    def get_true_pose(self) -> Tuple[float, float, float]:
        """Get ground truth robot pose."""
        return (self.true_x, self.true_y, self.true_theta)

    def _check_collision(self, x: float, y: float, walls: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> bool:
        """Check if robot would collide with any wall at given position."""
        for wall in walls:
            dist = self._point_to_line_distance(x, y, wall[0], wall[1])
            if dist < self.radius:
                return True
        return False

    def _point_to_line_distance(self, px: float, py: float, line_start: Tuple[float, float], line_end: Tuple[float, float]) -> float:
        """Calculate the distance from a point to a line segment."""
        x1, y1 = line_start
        x2, y2 = line_end

        line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)

        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_length**2)))
        
        closest_x = x1 + t * (x2 - x1)
        closest_y = y1 + t * (y2 - y1)
        
        return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

    def _line_intersection(self, line1_start: Tuple[float, float], line1_end: Tuple[float, float],
                          line2_start: Tuple[float, float], line2_end: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Calculate intersection point of two line segments."""
        x1, y1 = line1_start
        x2, y2 = line1_end
        x3, y3 = line2_start
        x4, y4 = line2_end

        denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denominator == 0:
            return None

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denominator

        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return (x, y)
        return None

class Environment:
    def __init__(self, width: int, height: int, **config):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Lidar Robot Simulation")
        self.clock = pygame.time.Clock()
        self.walls = self._create_walls()
        self.show_uncertainty = config.get('show_uncertainty', True)
        self.show_ground_truth = config.get('show_ground_truth', True)

    def _create_walls(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Create walls and obstacles in the environment."""
        walls = []
        # Outer walls
        walls.extend([
            ((0, 0), (12, 0)),      # Bottom
            ((0, 0), (0, 8)),       # Left
            ((12, 0), (12, 8)),     # Right
            ((0, 8), (12, 8))       # Top
        ])
        
        # Multiple obstacle groups
        # Group 1 - Bottom left
        walls.extend([
            ((2, 1), (3, 1)),       # Horizontal wall
            ((2, 1), (2, 2)),       # Vertical wall
            ((3, 1), (3, 2)),       # Vertical wall
            ((2, 2), (3, 2)),       # Horizontal wall
        ])
        
        # Group 2 - Bottom right
        walls.extend([
            ((9, 1), (10, 1)),      # Horizontal wall
            ((9, 1), (9, 2)),       # Vertical wall
            ((10, 1), (10, 2)),     # Vertical wall
            ((9, 2), (10, 2)),      # Horizontal wall
        ])
        
        # Group 3 - Top left
        walls.extend([
            ((2, 6), (3, 6)),       # Horizontal wall
            ((2, 6), (2, 7)),       # Vertical wall
            ((3, 6), (3, 7)),       # Vertical wall
            ((2, 7), (3, 7)),       # Horizontal wall
        ])
        
        # Group 4 - Top right
        walls.extend([
            ((9, 6), (10, 6)),      # Horizontal wall
            ((9, 6), (9, 7)),       # Vertical wall
            ((10, 6), (10, 7)),     # Vertical wall
            ((9, 7), (10, 7)),      # Horizontal wall
        ])
        
        # Central obstacles
        walls.extend([
            # Central pillar
            ((5.5, 3.5), (6.5, 3.5)), # Bottom
            ((5.5, 3.5), (5.5, 4.5)), # Left
            ((6.5, 3.5), (6.5, 4.5)), # Right
            ((5.5, 4.5), (6.5, 4.5)), # Top
            
            # Additional central obstacles
            ((4, 3), (5, 3)),         # Horizontal wall 1
            ((7, 3), (8, 3)),         # Horizontal wall 2
            ((4, 5), (5, 5)),         # Horizontal wall 3
            ((7, 5), (8, 5)),         # Horizontal wall 4
            
            # Diagonal walls
            ((3, 3), (4, 4)),         # Diagonal 1
            ((8, 4), (9, 3)),         # Diagonal 2
            ((3, 5), (4, 4)),         # Diagonal 3
            ((8, 4), (9, 5)),         # Diagonal 4
            
            # Small obstacles
            ((4, 2), (4.5, 2)),       # Small wall 1
            ((7.5, 2), (8, 2)),       # Small wall 2
            ((4, 6), (4.5, 6)),       # Small wall 3
            ((7.5, 6), (8, 6)),       # Small wall 4
        ])
        
        return walls

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int(x * self.width / 12)  # Adjusted for new map width
        screen_y = int(self.height - y * self.height / 8)  # Adjusted for new map height
        return (screen_x, screen_y)

    def draw(self, robots: List[LidarRobot]) -> None:
        """Draw the environment and robots."""
        self.screen.fill((255, 255, 255))  # White background
        
        # Draw walls
        for wall in self.walls:
            start = self.world_to_screen(wall[0][0], wall[0][1])
            end = self.world_to_screen(wall[1][0], wall[1][1])
            pygame.draw.line(self.screen, (0, 0, 0), start, end, 2)
        
        # Draw robots and their lidar readings
        for robot in robots:
            # Draw robot position and direction
            pos = self.world_to_screen(robot.x, robot.y)
            direction = (
                pos[0] + int(robot.radius * self.width/12 * math.cos(robot.theta)),
                pos[1] - int(robot.radius * self.width/12 * math.sin(robot.theta))
            )
            pygame.draw.circle(self.screen, robot.color, pos, int(robot.radius * self.width/12))
            pygame.draw.line(self.screen, (0, 0, 0), pos, direction, 2)
            
            # Draw ground truth if enabled
            if self.show_ground_truth:
                true_pos = self.world_to_screen(robot.true_x, robot.true_y)
                true_direction = (
                    true_pos[0] + int(robot.radius * self.width/12 * math.cos(robot.true_theta)),
                    true_pos[1] - int(robot.radius * self.width/12 * math.sin(robot.true_theta))
                )
                pygame.draw.circle(self.screen, (0, 0, 0), true_pos, int(robot.radius * self.width/12), 1)
                pygame.draw.line(self.screen, (0, 0, 0), true_pos, true_direction, 1)
            
            # Draw lidar readings
            if robot.lidar_readings:
                angles = np.linspace(-robot.lidar_angle_range/2, robot.lidar_angle_range/2, 
                                   len(robot.lidar_readings))
                for i, (angle, reading, uncertainty) in enumerate(zip(angles, robot.lidar_readings, robot.lidar_uncertainties)):
                    ray_angle = math.radians(angle) + robot.theta
                    end_x = robot.x + reading * math.cos(ray_angle)
                    end_y = robot.y + reading * math.sin(ray_angle)
                    end_pos = self.world_to_screen(end_x, end_y)
                    
                    # Draw lidar ray
                    pygame.draw.line(self.screen, (200, 200, 200), pos, end_pos, 1)
                    
                    # Draw uncertainty cone if enabled
                    if self.show_uncertainty and uncertainty < float('inf'):
                        uncertainty_angle = math.atan2(uncertainty, reading)
                        cone_angle1 = ray_angle - uncertainty_angle
                        cone_angle2 = ray_angle + uncertainty_angle
                        cone_end1 = (
                            robot.x + reading * math.cos(cone_angle1),
                            robot.y + reading * math.sin(cone_angle1)
                        )
                        cone_end2 = (
                            robot.x + reading * math.cos(cone_angle2),
                            robot.y + reading * math.sin(cone_angle2)
                        )
                        cone_points = [
                            pos,
                            self.world_to_screen(cone_end1[0], cone_end1[1]),
                            self.world_to_screen(cone_end2[0], cone_end2[1])
                        ]
                        pygame.draw.polygon(self.screen, (255, 200, 200, 128), cone_points, 0)
        
        pygame.display.flip()

    def get_walls(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get list of walls in the environment."""
        return self.walls

    def toggle_uncertainty(self) -> None:
        """Toggle visualization of sensor uncertainty."""
        self.show_uncertainty = not self.show_uncertainty

    def toggle_ground_truth(self) -> None:
        """Toggle visualization of ground truth."""
        self.show_ground_truth = not self.show_ground_truth 