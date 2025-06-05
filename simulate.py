import pygame
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from definitions import (
    SIMULATION_CONFIG, ROBOT_CONFIG, ENVIRONMENT_CONFIG,
    LidarRobot, Environment, DataRecorder
)


def main():
    # Initialize environment
    env = Environment(
        width=ENVIRONMENT_CONFIG['width'],
        height=ENVIRONMENT_CONFIG['height']
    )
    
    # Initialize data recorder
    data_recorder = DataRecorder('robot_data.npy')
    
    # Create two robots with different colors
    robot1 = LidarRobot(
        x=1.0, y=1.0, theta=0,  # Clear area in bottom-left
        color=ENVIRONMENT_CONFIG['colors']['robot1'],
        **ROBOT_CONFIG
    )
    robot2 = LidarRobot(
        x=11.0, y=7.0, theta=math.pi,  # Clear area in top-right
        color=ENVIRONMENT_CONFIG['colors']['robot2'],
        **ROBOT_CONFIG
    )
    robots = [robot1, robot2]
    
    running = True
    dt = SIMULATION_CONFIG['dt']
    frame_count = 0
    print_interval = 10  # Print every 10 frames to avoid flooding the console

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Save data before quitting
                data_recorder.save_data()
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_u:
                    env.toggle_uncertainty()
                elif event.key == pygame.K_g:
                    env.toggle_ground_truth()
                elif event.key == pygame.K_s:
                    # Save data when 's' key is pressed
                    data_recorder.save_data()

        # Handle keyboard input for robot 1 (WASD)
        keys = pygame.key.get_pressed()
        robot1.velocity = 0
        robot1.angular_velocity = 0
        
        if keys[pygame.K_w]:
            robot1.velocity = ROBOT_CONFIG['max_velocity']
        if keys[pygame.K_s]:
            robot1.velocity = -ROBOT_CONFIG['max_velocity']
        if keys[pygame.K_a]:
            robot1.angular_velocity = ROBOT_CONFIG['max_angular_velocity']
        if keys[pygame.K_d]:
            robot1.angular_velocity = -ROBOT_CONFIG['max_angular_velocity']

        # Handle keyboard input for robot 2 (Arrow keys)
        robot2.velocity = 0
        robot2.angular_velocity = 0
        
        if keys[pygame.K_UP]:
            robot2.velocity = ROBOT_CONFIG['max_velocity']
        if keys[pygame.K_DOWN]:
            robot2.velocity = -ROBOT_CONFIG['max_velocity']
        if keys[pygame.K_LEFT]:
            robot2.angular_velocity = ROBOT_CONFIG['max_angular_velocity']
        if keys[pygame.K_RIGHT]:
            robot2.angular_velocity = -ROBOT_CONFIG['max_angular_velocity']

        # Update robot states
        for robot in robots:
            robot.update(dt, env.get_walls())
            robot.get_lidar_readings(env.get_walls())
        
        # Record data for both robots
        angles = np.linspace(-ROBOT_CONFIG['lidar_angle_range']/2, 
                           ROBOT_CONFIG['lidar_angle_range']/2,
                           int(ROBOT_CONFIG['lidar_angle_range']/ROBOT_CONFIG['lidar_resolution']))
        
        data_recorder.record_data('robot1', robot1.get_true_pose(),
                                robot1.lidar_readings, angles, frame_count)
        data_recorder.record_data('robot2', robot2.get_true_pose(),
                                robot2.lidar_readings, angles, frame_count)
        
        # Print sensor data and pose every few frames
        # if frame_count % print_interval == 0:
        #     print("\n=== Robot 1 ===")
        #     print(f"Estimated Pose: {robot1.get_pose()}")
        #     print(f"Ground Truth Pose: {robot1.get_true_pose()}")
        #     print(f"LiDAR Readings: {robot1.lidar_readings[:5]}...")  # Show first 5 readings
        #     print(f"LiDAR Uncertainties: {robot1.lidar_uncertainties[:5]}...")  # Show first 5 uncertainties
            
        #     print("\n=== Robot 2 ===")
        #     print(f"Estimated Pose: {robot2.get_pose()}")
        #     print(f"Ground Truth Pose: {robot2.get_true_pose()}")
        #     print(f"LiDAR Readings: {robot2.lidar_readings[:5]}...")  # Show first 5 readings
        #     print(f"LiDAR Uncertainties: {robot2.lidar_uncertainties[:5]}...")  # Show first 5 uncertainties
        #     print("\n" + "="*50)
        
        frame_count += 1
        
        # Draw everything
        env.draw(robots)
        
        # Cap the frame rate
        env.clock.tick(SIMULATION_CONFIG['fps'])

    pygame.quit()

if __name__ == "__main__":
    main() 