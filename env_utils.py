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
    'lidar_range': 5.0,
    'lidar_angle_range': 180,
    'lidar_resolution': 2, 
    'radius': 0.05,  # Increased bot size for better visibility
    'sensor_noise': {
        'range_std': 0.0,  # Disabled range noise
        'angle_std': 0.0,  # Disabled angle noise
        'dropout_prob': 0.0,  # Disabled dropouts
        'min_range': 0.1,
        'max_range': 5.0
    },
    'motion_noise': {
        'pos_std': 0.0,  # Disabled velocity noise
        'theta_std': 0.0,  # Disabled angular noise
        'slip_prob': 0.0  # Disabled wheel slip
    }
}

ENVIRONMENT_CONFIG = {
    'width': 900,  # Reduced window width for better fit
    'height': 600,  # Reduced window height for better fit
    'x_width': 30,
    'y_height': 20,
    'wall_thickness': 0.1,  # Wall thickness in meters (must match mapping_utils default)
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
        velocity = self.velocity
        angular_velocity = self.angular_velocity
        
        # Simulate wheel slip
        if np.random.random() < self.motion_noise.get('slip_prob', 0):
            velocity *= 0.5
            angular_velocity *= 0.5
        
        # Calculate new position
        new_x = self.true_x + velocity * math.cos(self.true_theta) * dt
        new_y = self.true_y + velocity * math.sin(self.true_theta) * dt
        new_theta = self.true_theta + angular_velocity * dt
        new_theta = math.atan2(math.sin(new_theta), math.cos(new_theta))

        # Check for collisions with walls
        if not self._check_collision(new_x, new_y, walls):
            self.true_x = new_x
            self.true_y = new_y
            self.true_theta = new_theta
            
            # Update noisy pose estimate
            self.x = self.true_x + np.random.normal(0, self.motion_noise.get('pos_std', 0))
            self.y = self.true_y + np.random.normal(0, self.motion_noise.get('pos_std', 0))
            self.theta = self.true_theta + np.random.normal(0, self.motion_noise.get('theta_std', 0))
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
            
            # Find closest intersection with walls (accounting for wall thickness)
            min_distance = self.lidar_range
            wall_thickness = ENVIRONMENT_CONFIG['wall_thickness']  # Use centralized config
            
            for wall in walls:
                # Get the four edges of the thick wall rectangle
                thick_wall_edges = self._get_thick_wall_edges(wall[0], wall[1], wall_thickness)
                
                # Check intersection with each edge of the thick wall
                for edge in thick_wall_edges:
                    intersection = self._line_intersection(
                        (self.true_x, self.true_y), (ray_end_x, ray_end_y),
                        edge[0], edge[1]
                    )
                    if intersection:
                        distance = math.sqrt((intersection[0] - self.true_x)**2 + 
                                           (intersection[1] - self.true_y)**2)
                        min_distance = min(min_distance, distance)
                
            
            # Add range noise and handle dropouts
            if np.random.random() < self.sensor_noise.get('dropout_prob', 0):
                readings.append(self.lidar_range)
                uncertainties.append(float('inf'))
            elif min_distance < self.lidar_range:
                noisy_distance = min_distance + np.random.normal(0, self.sensor_noise.get('range_std', 0))
                noisy_distance = max(self.sensor_noise.get('min_range', 0.1),
                                   min(self.sensor_noise.get('max_range', 5.0), noisy_distance))
                readings.append(noisy_distance)
                uncertainty = (self.sensor_noise.get('range_std', 0) * 
                             (1 + min_distance / self.lidar_range))
                uncertainties.append(uncertainty)

            # If no obstacle is detected, add the max range
            else:
                readings.append(self.lidar_range)
                uncertainties.append(float(0))
                

        self.lidar_readings = readings
        self.lidar_uncertainties = uncertainties
        return readings, uncertainties

    def _get_thick_wall_edges(self, start: Tuple[float, float], end: Tuple[float, float], 
                             thickness: float) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get the four edges of a thick wall rectangle."""
        start_x, start_y = start
        end_x, end_y = end
        
        # Calculate wall direction vector
        wall_dx = end_x - start_x
        wall_dy = end_y - start_y
        wall_length = math.sqrt(wall_dx**2 + wall_dy**2)
        
        if wall_length == 0:
            return []  # Skip zero-length walls
        
        # Normalize direction vector
        wall_dx /= wall_length
        wall_dy /= wall_length
        
        # Calculate perpendicular vector for thickness
        perp_dx = -wall_dy
        perp_dy = wall_dx
        
        # Calculate half-thickness offset
        half_thickness = thickness / 2
        
        # Calculate the four corners of the thick wall rectangle
        corners = []
        
        # Top-left corner (start + perpendicular offset)
        corners.append((
            start_x + perp_dx * half_thickness,
            start_y + perp_dy * half_thickness
        ))
        
        # Top-right corner (start - perpendicular offset)
        corners.append((
            start_x - perp_dx * half_thickness,
            start_y - perp_dy * half_thickness
        ))
        
        # Bottom-right corner (end - perpendicular offset)
        corners.append((
            end_x - perp_dx * half_thickness,
            end_y - perp_dy * half_thickness
        ))
        
        # Bottom-left corner (end + perpendicular offset)
        corners.append((
            end_x + perp_dx * half_thickness,
            end_y + perp_dy * half_thickness
        ))
        
        # Return the four edges of the rectangle
        return [
            (corners[0], corners[1]),  # Top edge
            (corners[1], corners[2]),  # Right edge
            (corners[2], corners[3]),  # Bottom edge
            (corners[3], corners[0])   # Left edge
        ]

    def get_pose(self) -> Tuple[float, float, float]:
        """Get current robot pose (x, y, theta)."""
        return (self.x, self.y, self.theta)

    def get_true_pose(self) -> Tuple[float, float, float]:
        """Get ground truth robot pose."""
        return (self.true_x, self.true_y, self.true_theta)

    def _check_collision(self, x: float, y: float, walls: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> bool:
        """Check if robot would collide with any wall at given position."""
        wall_thickness = ENVIRONMENT_CONFIG['wall_thickness']  # Use centralized config
        
        for wall in walls:
            # Check collision with thick wall by checking distance to all edges
            thick_wall_edges = self._get_thick_wall_edges(wall[0], wall[1], wall_thickness)
            
            for edge in thick_wall_edges:
                dist = self._point_to_line_distance(x, y, edge[0], edge[1])
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
    def __init__(self, width: int, height: int, screen=None, **config):
        self.width = width
        self.height = height
        self.x_width = ENVIRONMENT_CONFIG['x_width']
        self.y_height = ENVIRONMENT_CONFIG['y_height']
        self.screen = screen if screen is not None else pygame.display.set_mode((width, height))
        if screen is None:
            pygame.display.set_caption("Lidar Robot Simulation")
        self.clock = pygame.time.Clock()
        self.walls = self._create_walls()
        self.show_uncertainty = config.get('show_uncertainty', False)
        self.show_ground_truth = config.get('show_ground_truth', True)

    def _create_walls(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        xw = self.x_width
        yh = self.y_height
        walls = []
        # Outer walls
        walls.extend([
            ((0, 0), (xw, 0)),      # Bottom
            ((0, 0), (0, yh)),      # Left
            ((xw, 0), (xw, yh)),    # Right
            ((0, yh), (xw, yh))     # Top
        ])
        # Long horizontal shelves (spread across the map)
        # walls.append(((0.17*xw, 0.35*yh), (0.83*xw, 0.35*yh)))    # Mid left
        walls.append(((0.83*xw, 0.35*yh), (0.83*xw, 0.65*yh)))   # Mid right
        walls.append(((0.10*xw, 0.55*yh), (0.43*xw, 0.55*yh)))  # Upper left
        walls.append(((0.57*xw, 0.55*yh), (0.90*xw, 0.55*yh))) # Upper right
        walls.append(((0.20*xw, 0.10*yh), (0.20*xw, 0.40*yh)))     # Lower left
        walls.append(((0.80*xw, 0.10*yh), (0.80*xw, 0.40*yh)))   # Lower right
        walls.append(((0.40*xw, 0.25*yh), (0.40*xw, 0.60*yh)))  # Mid left
        walls.append(((0.10*xw, 0.70*yh), (0.10*xw, 0.90*yh)))   # Far left
        walls.append(((0.90*xw, 0.70*yh), (0.90*xw, 0.90*yh))) # Far right
        # Diagonal walls
        walls.append(((0.27*xw, 0.20*yh), (0.40*xw, 0.30*yh)))    # Lower diagonal
        walls.append(((0.80*xw, 0.60*yh), (0.93*xw, 0.70*yh))) # Upper diagonal
        walls.append(((0.20*xw, 0.80*yh), (0.33*xw, 0.90*yh)))  # Top diagonal
        walls.append(((0.67*xw, 0.80*yh), (0.80*xw, 0.90*yh))) # Top diagonal
        # L-shaped shelves
        walls.append(((0.07*xw, 0.10*yh), (0.20*xw, 0.10*yh)))     # L1 horizontal
        walls.append(((0.07*xw, 0.10*yh), (0.07*xw, 0.30*yh)))     # L1 vertical
        walls.append(((0.80*xw, 0.10*yh), (0.93*xw, 0.10*yh)))   # L2 horizontal
        walls.append(((0.93*xw, 0.10*yh), (0.93*xw, 0.30*yh)))   # L2 vertical
        # T-shaped shelves
        walls.append(((0.47*xw, 0.25*yh), (0.60*xw, 0.25*yh)))   # T1 top
        walls.append(((0.53*xw, 0.25*yh), (0.53*xw, 0.45*yh)))   # T1 stem
        walls.append(((0.27*xw, 0.45*yh), (0.40*xw, 0.45*yh)))    # T2 top
        walls.append(((0.33*xw, 0.45*yh), (0.33*xw, 0.65*yh)))  # T2 stem
        walls.append(((0.67*xw, 0.65*yh), (0.80*xw, 0.65*yh))) # T3 top
        walls.append(((0.73*xw, 0.65*yh), (0.73*xw, 0.85*yh))) # T3 stem
        # Zig-zag walls
        walls.append(((0.13*xw, 0.25*yh), (0.20*xw, 0.25*yh)))     # Zig1
        walls.append(((0.20*xw, 0.25*yh), (0.20*xw, 0.35*yh)))     # Zig1
        walls.append(((0.20*xw, 0.35*yh), (0.27*xw, 0.35*yh)))     # Zig1
        walls.append(((0.27*xw, 0.35*yh), (0.27*xw, 0.45*yh)))     # Zig1
        walls.append(((0.87*xw, 0.35*yh), (0.87*xw, 0.45*yh)))   # Zig2
        walls.append(((0.87*xw, 0.45*yh), (0.97*xw, 0.45*yh)))   # Zig2
        # Pallets (scaled)
        pallet_centers = [
            (0.13*xw, 0.05*yh), (0.27*xw, 0.05*yh), (0.40*xw, 0.05*yh), (0.53*xw, 0.05*yh), (0.67*xw, 0.05*yh), (0.80*xw, 0.05*yh), (0.97*xw, 0.05*yh),
            (0.10*xw, 0.20*yh), (0.23*xw, 0.20*yh), (0.37*xw, 0.20*yh), (0.50*xw, 0.20*yh), (0.63*xw, 0.20*yh), (0.77*xw, 0.20*yh), (0.90*xw, 0.20*yh),
            (0.17*xw, 0.30*yh), (0.30*xw, 0.30*yh), (0.43*xw, 0.30*yh), (0.57*xw, 0.30*yh), (0.70*xw, 0.30*yh), (0.83*xw, 0.30*yh), (0.97*xw, 0.30*yh),
            (0.07*xw, 0.45*yh), (0.20*xw, 0.45*yh), (0.33*xw, 0.45*yh), (0.47*xw, 0.45*yh), (0.60*xw, 0.45*yh), (0.73*xw, 0.45*yh), (0.87*xw, 0.45*yh),
            (0.10*xw, 0.80*yh), (0.23*xw, 0.80*yh), (0.37*xw, 0.80*yh), (0.50*xw, 0.80*yh), (0.63*xw, 0.80*yh), (0.77*xw, 0.80*yh), (0.90*xw, 0.80*yh),
            (0.17*xw, 0.90*yh), (0.30*xw, 0.90*yh), (0.43*xw, 0.90*yh), (0.57*xw, 0.90*yh), (0.70*xw, 0.90*yh), (0.83*xw, 0.90*yh), (0.97*xw, 0.90*yh),
            (0.05*xw, 0.15*yh), (0.18*xw, 0.40*yh), (0.32*xw, 0.65*yh), (0.45*xw, 0.85*yh), (0.58*xw, 0.10*yh), (0.72*xw, 0.35*yh), (0.85*xw, 0.60*yh), (0.98*xw, 0.80*yh),
            (0.08*xw, 0.30*yh), (0.22*xw, 0.55*yh), (0.35*xw, 0.75*yh), (0.48*xw, 0.95*yh), (0.62*xw, 0.20*yh), (0.75*xw, 0.45*yh), (0.88*xw, 0.70*yh), (0.95*xw, 0.40*yh)
        ]
        pallet_size = 0.5 * (xw/30)  # Scaled with xw
        for cx, cy in pallet_centers:
            half = pallet_size / 2
            walls.extend([
                ((cx - half, cy - half), (cx + half, cy - half)),
                ((cx + half, cy - half), (cx + half, cy + half)),
                ((cx + half, cy + half), (cx - half, cy + half)),
                ((cx - half, cy + half), (cx - half, cy - half)),
            ])
        # Circular columns (approximated as octagons) - more spread out
        column_centers = [(0.23*xw, 0.25*yh), (0.77*xw, 0.25*yh), (0.37*xw, 0.50*yh),  (0.17*xw, 0.65*yh), (0.90*xw, 0.65*yh)]
        column_radius = 0.4 * (xw/30)
        for cx, cy in column_centers:
            points = []
            for i in range(8):
                angle = i * (2 * math.pi / 8)
                px = cx + column_radius * math.cos(angle)
                py = cy + column_radius * math.sin(angle)
                points.append((px, py))
            for i in range(8):
                walls.append((points[i], points[(i+1)%8]))
        return walls
    
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int(x * self.width / self.x_width)  # Adjusted for new map width
        screen_y = int(self.height - y * self.height / self.y_height)  # Adjusted for new map height
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
                pos[0] + int(robot.radius * self.width/self.x_width * math.cos(robot.theta)),
                pos[1] - int(robot.radius * self.height/self.y_height * math.sin(robot.theta))
            )
            pygame.draw.circle(self.screen, robot.color, pos, int(robot.radius * self.width/self.x_width))
            pygame.draw.line(self.screen, (0, 0, 0), pos, direction, 2)
            
            # Draw ground truth if enabled
            if self.show_ground_truth:
                true_pos = self.world_to_screen(robot.true_x, robot.true_y)
                true_direction = (
                    true_pos[0] + int(robot.radius * self.width/self.x_width * math.cos(robot.true_theta)),
                    true_pos[1] - int(robot.radius * self.height/self.y_height * math.sin(robot.true_theta))
                )
                pygame.draw.circle(self.screen, (0, 0, 0), true_pos, int(robot.radius * self.width/self.x_width), 1)
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

class DataRecorder:
    def __init__(self, filename: str):
        self.filename = filename
        self.data = {
            'robot1': {
                'poses': [],  # List of (x, y, theta) tuples
                'lidar_readings': [],  # List of (range, bearing) tuples
                'timestamps': []  # List of frame numbers
            },
            'robot2': {
                'poses': [],
                'lidar_readings': [],
                'timestamps': []
            }
        }
        self.movement_threshold = 0.1  # 5cm movement threshold
        self.angle_threshold = 0.1 # ~5.7 degrees rotation threshold
        self.last_poses = {
            'robot1': None,
            'robot2': None
        }

    def should_record(self, robot_id: str, current_pose: Tuple[float, float, float]) -> bool:
        """Check if we should record data based on movement threshold."""
        if self.last_poses[robot_id] is None:
            self.last_poses[robot_id] = current_pose
            return True

        last_x, last_y, last_theta = self.last_poses[robot_id]
        current_x, current_y, current_theta = current_pose

        # Calculate movement distance
        movement = math.sqrt((current_x - last_x)**2 + (current_y - last_y)**2)
        
        # Calculate angle difference
        angle_diff = abs(math.atan2(math.sin(current_theta - last_theta),
                                  math.cos(current_theta - last_theta)))

        if movement > self.movement_threshold or angle_diff > self.angle_threshold:
            self.last_poses[robot_id] = current_pose
            return True
        return False

    def record_data(self, robot_id: str, pose: Tuple[float, float, float],
                   lidar_readings: List[float], lidar_angles: List[float],
                   timestamp: int) -> None:
        """Record robot data if movement threshold is met."""
        if self.should_record(robot_id, pose):
            self.data[robot_id]['poses'].append(pose)
            self.data[robot_id]['timestamps'].append(timestamp)
            
            # Convert lidar readings to (range, bearing) pairs
            lidar_data = list(zip(lidar_readings, lidar_angles))
            self.data[robot_id]['lidar_readings'].append(lidar_data)

    def save_data(self) -> None:
        """Save recorded data to a .npy file."""
        np.save(self.filename, self.data)
        print(f"Data saved to {self.filename}") 