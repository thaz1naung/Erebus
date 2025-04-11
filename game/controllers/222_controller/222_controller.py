# ======================
# STANDARD LIBRARIES
# ======================
import os
import re
import time
import math
import heapq
import random

# ======================
# THIRD-PARTY LIBRARIES
# ======================
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
import sklearn
import easyocr
import json

# ======================
# WEBOTS CONTROLLER
# ======================
from controller import Robot
# from shape_detection_controller import Shape  

# ======================
# ENVIRONMENT SETTINGS
# ======================
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable OneDNN for TensorFlow 

# ======================
# SIMULATION CONSTANTS
# ======================
TIME_STEP = 32
MAP_SIZE = 8           # Grid size (NxN)
CELL_SIZE = 0.2        # Size of each grid cell in meters
CAMERA_NAME = "camera_centre"
TURN_DURATION = 40     # Steps to complete 90-degree turn

# ======================
# MOTION PARAMETERS
# ======================
FORWARD_SPEED = 3.0
TURN_SPEED = 0.5

# ======================
# SIGN DETECTION THRESHOLDS
# ======================
MIN_AREA = 2000
MAX_AREA = 1077000
MODEL_INPUT_SIZE = (128, 128)
STOP_DISTANCE = 300    # Sign area threshold to stop

# ======================
# TOLERANCE SETTINGS
# ======================
TOLERANCE_RADIUS = 1       # Grid cell matching radius
COLOR_TOLERANCE = 15       # RGB tolerance for tile detection

# ======================
# TRACKING DATA STRUCTURES
# ======================
processed_signs = set()
processed_sign_bounding_boxes = []
processed_sign_classes = {}
processed_tiles = {}        # Map of tiles with grid positions

# ======================
# LOAD OCR AND CLASSIFIER
# ======================
model_path = "text_classifier-2.pkl"
pipeline = joblib.load(model_path)           # Trained text classification pipeline
reader = easyocr.Reader(['en'])              # Initialize EasyOCR

# ======================
# SIGN CLASS MAPPING (TEXT → LABEL)
# ======================
class_to_letter = {
    "flammable": "F",
    "corrosive": "C",
    "H": "H",
    "organic": "O",
    "poison": "P",
    "S": "S",
    "U": "U"
}

# ======================
# TILE COLOR MAPPING (R, G, B, Number)
# ======================
TILE_MAPPING = {
    "black": (60, 60, 60, 2),
    "swamp": (142, 222, 245, 3),
    "purple": (251, 91, 193, 7),
    "blue": (255, 91, 91, 6),
    "grey": (124, 117, 115, 2),
    "green": (48, 255, 48, 9),
    "red": (91, 91, 255, 8),
    "checkpoint": (100, 93, 91, 4)
}   
                    
# Robot initialization                            

class MyRobotMovement(Robot):
    def __init__(self, verbose=True):
        """ Initializes the robot and the required devices. """
        super().__init__()
        self.MAX_SPEED = 3.5
        self.ANGLE_THRESHOLD = 1
        self.DISTANCE_THRESHOLD = 0.197
        self.DISTANCE_THRESHOLD_Z = 0.116
        self.DISTANCE_THRESHOLD_Y = 0.116 
        self.DISTANCE = 0.2
        self.timestep = 32
        self.verbose = verbose  # Detailed output on/off
        self.current_angle = 0
        self.detected_sign = None
        self.sign_classified = False
        self.target_cell = None 
        self.grid_size = 8  # Define grid size
        self.NORTH_WALL = 10
        self.SOUTH_WALL = 9
        self.EAST_WALL = 8
        self.WEST_WALL = 7
        self.VISITED = 6
        self.HAZG = 5
        self.CHECKPOINT = 4
        self.VICH = 3
        self.START_TILE = 2
        self.HAZS = 1
        self.LIDAR = 0
        self.initial = 0b00000000000  # Binary representation (same as 0)
        self.grid_map = {}
        self.grid =np.full((self.grid_size, self.grid_size), self.initial, dtype=int)# [[self.initial] * self.grid_size for _ in range(self.grid_size)]  # Fill the grid
        self.start_point = True
        self.runonce = 0
        self.WHEEL_RADIUS = 0.02  # In meters
        self.prev_left_pos = 0
        self.prev_right_pos = 0
        self.runonce_2 = 0

        self.processed_signs_dict = {}
        self.processed_tile_dict = {}


        # Get Display Node
        self.display = self.getDevice("display")  # Change name if needed
        if not self.display:
            print("Error: No Display Node found!")
            return

        # Display parameters
        self.display_width = self.display.getWidth()
        self.display_height = self.display.getHeight()
        self.center_x = self.display_width // 2
        self.center_y = self.display_height // 2

        
        # List to store previous positions
        self.path = []
        
    def initialize_devices(self):
        """ Initializes sensors and actuators. """
        # Sensors
        self.iu = self.getDevice('inertialunit')
        self.iu.enable(self.timestep)
        self.gps = self.getDevice('gps')
        self.gps.enable(self.timestep)
        
        self.lidar = self.getDevice('lidar')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud();
        
        self.color_sensor = self.getDevice('colour_sensor')
        self.color_sensor.enable(self.timestep)
        #self.camera = self.getDevice('camera')
        #self.camera.enable(self.timestep)
        self.ps = []
        self.psNames = ['ps0', 'ps1', 'ps2', 'ps3', 'distance sensor5', 'distance sensor6', 'distance sensor7','distance sensor8']
        #self.initial = 0b000000000000  # Binary representation (same as 0)
        
        #self.grid = [[self.initial] * self.grid_size for _ in range(self.grid_size)]  # Fill the grid
        for i in range(8):
            self.ps.append(self.getDevice(self.psNames[i]))
            self.ps[i].enable(self.timestep)
            
        self.front_sensor = self.getDevice('ps0')
        self.right_sensor = self.getDevice('distance sensor8')
        self.left_sensor = self.getDevice('distance sensor6')
        self.middle_right_sensor = self.getDevice('ps2')
        self.back_left_sensor = self.getDevice('distance sensor5')
        self.back_right_sensor = self.getDevice('ps3')
        
        self.camera = self.getDevice(CAMERA_NAME)
        self.camera.enable(TIME_STEP)

        self.sensors = [self.front_sensor, self.right_sensor, self.left_sensor, self.middle_right_sensor, self.back_left_sensor, self.back_right_sensor]
        for sensor in self.sensors:
            sensor.enable(self.timestep)
        # Some devices, such as the InertialUnit, need some time to "warm up"
        self.wait()
        # Actuators
        self.leftMotor = self.getDevice('wheel1 motor')
        self.rightMotor = self.getDevice('wheel2 motor')
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)
        
        self.left_position_sensor = self.getDevice("wheel1 sensor")
        self.right_position_sensor = self.getDevice("wheel2 sensor")
        self.left_position_sensor.enable(self.timestep)
        self.right_position_sensor.enable(self.timestep)

        # Background setup
        self.display.setColor(0xFFFFFF)  # Black background
        self.display.fillRectangle(0, 0, self.display_width, self.display_height)
        self.display.setColor(0xFF0000)  # Red color for path
        self.display.setColor(0xFF0000)  # Red color
        self.display.drawPixel(256, 256)  # Draw a pixel at (x=256, y=256)


        # **Log Initial Yaw**
        initial_yaw = self.get_yaw()
        print(f"🧭 Initial Robot Yaw: {initial_yaw}° ({self.get_robot_direction()})")
        # **Force alignment to EAST (0°) if necessary**
        if initial_yaw != 0:
            print("🔄 Rotating to EAST (0°) for consistent start position...")
            self.rotate_to(0)
            print(f"✅ Robot now facing: {self.get_robot_direction()}")        
    
    # Plot robot position on display node to show it in runtime
    def plot_position(self):
        gps_values = self.gps.getValues()  # Get current GPS (x, y, z)
        x, _, y = gps_values  # Ignore height (y-axis in Webots is vertical)

        # Convert world coordinates to display coordinates
        scaled_x = int(self.center_x + x * 50)  # Scale factor to fit on Display
        scaled_y = int(self.center_y + y * 50)

        # Append to path history
        self.path.append((scaled_x, scaled_y))

        # Redraw the path
        for i in range(1, len(self.path)):
            x1, y1 = self.path[i - 1]
            x2, y2 = self.path[i]
            self.display.drawLine(x1, y1, x2, y2)

    def wait(self):
        """ Waits for 500ms. """
        self.step(500)
        
    def get_yaw(self):
        """ Gets the yaw value from the InertialSensor, 
        converts it in degrees and normalizes it. """
        values = self.iu.getRollPitchYaw()
        yaw = round(math.degrees(values[2]))
        # The InertialUnit gives us a reading based on how the robot is oriented with
        # respect to the X axis of the world: EAST 0°, NORTH 90°, WEST 180°, SOUTH -90°.
        # This operation converts these values to a normal, positive circumfrence.
        if yaw < 0:
            yaw += 360
        return yaw
        
    def get_direction(self):
        """ Determines the robot's current facing direction based on yaw. """
        yaw = self.get_yaw()
        
        if 0 <= yaw < 45 or yaw >= 315:
            return "East"
        elif 45 <= yaw < 135:
            return "North"
        elif 135 <= yaw < 225:
            return "West"
        else:  # 225 <= yaw < 315
            return "South"
            
    # To generate a midpoint grid which is the midpoint of each cell based on the inital position of the robot.
    def generate_gps_grid(self, initial_gps, robotpos, grid_size=8, cell_size=0.115):

        x0, y0, z0 = (round(coord, 4) for coord in initial_gps)  # Ensure 4 decimal places

        gps_grid = [[None] * grid_size for _ in range(grid_size)]  # Initialize empty grid

        for row in range(grid_size):  
            for col in range(grid_size):  
                if robotpos == 'topleft':
                    x = x0 + (col * cell_size)  # Moving East (right)
                    z = z0 + (row * cell_size)  # Moving North (up)
                elif robotpos == 'bottomleft':
                    x = x0 - (row * cell_size)  # Moving East (right)
                    z = z0 - (col * cell_size)  # Moving North (up)
                elif robotpos == 'topright':
                    x = x0 - (col * cell_size)  # Moving East (right)
                    z = z0 - (row * cell_size)  # Moving North (up)
                elif robotpos == 'bottomright':
                    x = x0 + (row * cell_size)  # Moving East (right)
                    z = z0 + (col * cell_size)  # Moving North (up)
                else:
                    # Fallback to prevent crash if robotpos is invalid
                    x = x0 + (col * CELL_SIZE)
                    z = z0 + (row * CELL_SIZE)
                    
                gps_grid[row][col] = (round(x, 4), round(y0, 4), round(z, 4))  

        return gps_grid


    # Create a cell grid based on the midpoint of each cell

    def generate_shared_wall_grid(self, gps_grid, cell_size=0.115):

        grid_size = 2 * self.grid_size + 1  # Calculate final grid size
        grid = [[(1.0, 1.0, 1.0)] * grid_size for _ in range(grid_size)]  # Initialize everything as walls (1s)
        half_size = cell_size / 2

        for row in range(len(gps_grid)):
            for col in range(len(gps_grid)):
                mid_x, mid_y, mid_z = gps_grid[row][col]
                grid[2 * row + 1][2 * col + 1] = (mid_x, mid_y, mid_z)

                # Compute edge points
                corners = {
                    "top_left": (mid_x - half_size, mid_y, mid_z - half_size),
                    "top_right": (mid_x + half_size, mid_y, mid_z - half_size),
                    "bottom_left": (mid_x - half_size, mid_y, mid_z + half_size),
                    "bottom_right": (mid_x + half_size, mid_y, mid_z + half_size),
                }
                walls = {
                    "top": (mid_x, mid_y, mid_z - half_size),
                    "bottom": (mid_x, mid_y, mid_z + half_size),
                    "left": (mid_x - half_size, mid_y, mid_z),
                    "right": (mid_x + half_size, mid_y, mid_z),
                }

                # Assign corner points
                grid[2 * row][2 * col] = corners["top_left"]
                grid[2 * row][2 * col + 2] = corners["top_right"]
                grid[2 * row + 2][2 * col] = corners["bottom_left"]
                grid[2 * row + 2][2 * col + 2] = corners["bottom_right"]

                # Assign shared wall points
                grid[2 * row][2 * col + 1] = walls["top"]  # Top wall
                grid[2 * row + 2][2 * col + 1] = walls["bottom"]  # Bottom wall
                grid[2 * row + 1][2 * col] = walls["left"]  # Left wall
                grid[2 * row + 1][2 * col + 2] = walls["right"]  # Right wall

        return grid
        
    
    # Update the value of each cell as it reaches the next cell
    def update_dict(self, position, sensor_data, direction, current_gps):
        x = position[0]
        y = position[1]
        self.grid_map[(x, y)] = {
            "N": sensor_data[0],
            "S": sensor_data[2],
            "E": sensor_data[1],
            "W": sensor_data[3],
            "V": True,  # Mark the tile as visited
            "O": direction,
            "GPS": current_gps
        }

        

    # Distance sensors         
    def distance_sensors(self, sensor_id):
        psValues = []
        for i in range(8):
            psValues.append(robot.ps[i].getValue())
        return psValues[sensor_id]
        
    def get_sensor_data(self, current_direction):
        front_sensor = 1 if (self.distance_sensors(0) + self.distance_sensors(7)) < 0.23 else 0
        back_sensor = 1 if (self.distance_sensors(3) + self.distance_sensors(4)) < 0.23 else 0
        right_sensor = 1 if (self.distance_sensors(1) + self.distance_sensors(2)) < 0.23 else 0
        left_sensor = 1 if (self.distance_sensors(5) + self.distance_sensors(5)) < 0.23 else 0
        # can make this into a function by calling it for sensor data
        if current_direction == 'East':
            sensor_data = [left_sensor, front_sensor, right_sensor, back_sensor]
        elif current_direction == 'North':
            sensor_data = [front_sensor, right_sensor, back_sensor, left_sensor]
        elif current_direction == 'South':
            sensor_data = [back_sensor, left_sensor, front_sensor, right_sensor]
        elif current_direction == 'West':
            sensor_data = [right_sensor, back_sensor, left_sensor, front_sensor]


        return sensor_data, front_sensor, left_sensor, right_sensor
        
    # Function to update the occupancy grid in 11bit value
    def update_tile(self, position, sensor_data, direction, start_point, L_v, V_v, processed_tiles = 0):
        x = position[0]
        y = position[1]
        is_north_wall = True if sensor_data[0] == 1 else False
        is_east_wall = True if sensor_data[1] == 1 else False
        is_south_wall = True if sensor_data[2] == 1 else False
        is_west_wall = True if sensor_data[3] == 1 else False
        is_visited = True if V_v == 1 else False
        is_LidarV = True if L_v == 1 else False

        
        if robot.start_point == True:
            tile_status = 0b00001000100  # Initial state (optional)
            robot.start_point = False
        else:
            tile_status = self.grid[x][y]

        if self.processed_signs_dict:
            for key, value in self.processed_signs_dict.items():
                sign_posx = key[0]
                sign_posy = key[1]
                if x == sign_posx and y == sign_posy:
                    if value == 'F' or value == 'P':
                        bit_pos = self.HAZS
                    elif value == 'H' or value == 'S' or value == 'U':
                        bit_pos = self.VICH

                    if not self.is_bit_set(tile_status, bit_pos):
                            tile_status = self.set_bit(tile_status, bit_pos)
                            print(f'Bit set at 1 at ({sign_posx}, {sign_posy})')


        if self.processed_tile_dict:
            for key, value in self.processed_tile_dict.items():
                sign_posx = key[0]
                sign_posy = key[1]
                if x == sign_posx and y == sign_posy:
                    if value['type'] == 'swamp':
                        if not self.is_bit_set(tile_status, self.HAZG):
                            tile_status = self.set_bit(tile_status, self.HAZG)
                            print(f'Bit set at 1 for {value['type']} at ({sign_posx}, {sign_posy})')
                    elif value['type'] == 'checkpoint':
                        if not self.is_bit_set(tile_status, self.CHECKPOINT):
                            tile_status = self.set_bit(tile_status, self.CHECKPOINT)
                            print(f'Bit set at 1 for {value['type']} at ({sign_posx}, {sign_posy})')

        

        if not self.is_bit_set(tile_status, self.LIDAR) and not is_visited:
            
            if is_north_wall:
                tile_status = self.set_bit(tile_status, self.NORTH_WALL)

            if is_east_wall:
                tile_status = self.set_bit(tile_status, self.EAST_WALL)

            if is_south_wall:
                tile_status = self.set_bit(tile_status, self.SOUTH_WALL)

            if is_west_wall:
                tile_status = self.set_bit(tile_status, self.WEST_WALL)

            if is_LidarV:
                tile_status = self.set_bit(tile_status, self.LIDAR)

        elif is_visited and self.is_bit_set(tile_status, self.LIDAR):
            
            tile_status = self.set_bit(tile_status, self.VISITED)
            if is_north_wall:
                tile_status = self.set_bit(tile_status, self.NORTH_WALL)

            if is_east_wall:
                tile_status = self.set_bit(tile_status, self.EAST_WALL)

            if is_south_wall:
                tile_status = self.set_bit(tile_status, self.SOUTH_WALL)

            if is_west_wall:
                tile_status = self.set_bit(tile_status, self.WEST_WALL)

        elif V_v == 1:
            if is_north_wall:
                tile_status = self.set_bit(tile_status, self.NORTH_WALL)

            if is_east_wall:
                tile_status = self.set_bit(tile_status, self.EAST_WALL)

            if is_south_wall:
                tile_status = self.set_bit(tile_status, self.SOUTH_WALL)

            if is_west_wall:
                tile_status = self.set_bit(tile_status, self.WEST_WALL)

        self.grid[x][y] = tile_status



    # Function to set a bit
    def set_bit(self, value, bit_position):
        return value | (1 << bit_position)


    # Function to clear a bit (turn a feature OFF)
    def clear_bit(self, value, bit_position):
        return value & ~(1 << bit_position)

    # Function to check if a bit is set
    def is_bit_set(self, value, bit_position):
        return (value & (1 << bit_position)) != 0


    # Get lidar points
    def get_lidar_points(self, current_gps, current_direction, yaw1):
        gps_x_point_1 = []
        gps_y_point_1 = []
        combined_gps_points = []

        # Transform LiDAR points to global coordinates
        rangeImagemain = robot.lidar.getRangeImage()
        rangeImage = rangeImagemain[1024:1536]
        pointCloudImage = robot.lidar.getPointCloud()
        pointCloudImage = pointCloudImage[1024:1536]

        for point in pointCloudImage:
            if current_direction == 'East' or current_direction == 'West':
                global_x = current_gps[0] + (point.x * math.sin(yaw1) + point.y * math.cos(yaw1))
                global_y = current_gps[2] + (point.x * math.cos(yaw1) - point.y * math.sin(yaw1))
            elif current_direction == 'South':
                global_x = current_gps[0] + (point.x * math.sin(yaw1) - point.y * math.cos(yaw1))
                global_y = current_gps[2] + (point.x * math.cos(yaw1) - point.y * math.sin(yaw1))
            else:
                global_x = current_gps[0] + (point.x * math.sin(yaw1) - point.y * math.cos(yaw1))
                global_y = current_gps[2] + (point.x * math.cos(yaw1) - point.y * math.sin(yaw1))

            gps_x_point_1.append(global_x)
            gps_y_point_1.append(global_y)

        for i in range(len(gps_x_point_1)):
            combined_gps_points.append([gps_x_point_1[i], gps_y_point_1[i]])

        return combined_gps_points, gps_x_point_1, gps_y_point_1


    # Convert the lidar points to check if the points overlap with the cellgrid
    def convert_lidar_points(self, combined_gps_points, gps_grid, cell_grid, current_direction):
        filtered_lidar_points = [point for point in combined_gps_points if all(math.isfinite(coord) for coord in point)]

        # print(cell_grid)
        grid_1 = gps_grid

        grid_2 = cell_grid

        # List of lists to store the mapping
        mapped_grid = []

        # Loop through grid_1 and extract corresponding 3x3 blocks from grid_2
        for i in range(len(grid_1)):  # 4 rows
            row_blocks = []  # Store one row of mapped 3x3 blocks
            for j in range(len(grid_1[i])):  # 4 cols (corrected)
                # Calculate the row and column ranges dynamically based on the block size
                start_row = 2 * i  # Start row is based on the equation 2n + 1
                end_row = start_row + 3
                start_col = 2 * j  # Start column is based on the equation 2n + 1
                end_col = start_col + 3

                # Make sure we don't go out of bounds, handle edge cases
                block = [row[start_col:end_col] for row in grid_2[start_row:end_row]]
                row_blocks.append(block)  # Append the 3x3 block

            mapped_grid.append(row_blocks)  # Append the row of blocks

       
        lidar_points = filtered_lidar_points


        threshold = 0.04  # Allowable difference in degrees
        

        lidar_grid = [[[0, 0, 0, 0, 0] for _ in range(len(mapped_grid))] for _ in range(len(mapped_grid))]
        for a in range(len(mapped_grid)):
            for b in range(len(mapped_grid)):
                world_points = mapped_grid[a][b]

                count_n = np.zeros(3)
                count_s = np.zeros(3)
                count_e = np.zeros(3)
                count_w = np.zeros(3)
                for k in range(len(lidar_points)):
                    row_count = np.zeros(3)
                  
                    for m, i in enumerate(world_points):
                 
                        for l in range(3):
                            if abs(lidar_points[k][0] -  i[l][0]) <= threshold:
                                if abs(lidar_points[k][1] - i[l][2]) <= threshold:
                                    if m == 0:
                                        count_e[l] += 1
                                    elif m == 2:
                                        count_w[l] += 1


                        if abs(lidar_points[k][0] -  i[0][0]) <= threshold:
                            if abs(lidar_points[k][1] -  i[0][2]) <= threshold:
                                count_n[m] += 1 
                        
                        if abs(lidar_points[k][0] -  i[2][0]) <= threshold:
                            if abs(lidar_points[k][1] -  i[2][2]) <= threshold:
                                count_s[m] += 1 
                        
                N_v = S_v = E_v = W_v = L_v = 0
                if sum(1 for var in [count_n[0], count_n[1], count_n[2]] if var) < 2:
                    N_v = 0
                else:
                    if count_n[1] == 0.0:
                        N_v = 0
                    else:
                        N_v = L_v = 1
               
                if sum(1 for var in [count_s[0], count_s[1], count_s[2]] if var) < 2:
                    S_v = 0
                else:
                    if count_s[1] == 0.0:
                        S_v = 0
                    else:
                        S_v = L_v = 1

                if sum(1 for var in [count_e[0], count_e[1], count_e[2]] if var) < 2:
                    E_v = 0
                else:
                    if count_e[1] == 0.0:
                        E_v = 0
                    else:
                        E_v = L_v = 1

                if sum(1 for var in [count_w[0], count_w[1], count_w[2]] if var) < 2:
                    W_v = 0
                else:
                    if count_w[1] == 0.0:
                        W_v = 0
                    else:
                        W_v = L_v = 1


                lidar_grid[a][b] = [N_v, S_v, E_v, W_v, L_v]
                
                sensor_data= [N_v, E_v, S_v, W_v]
                start_position = (a, b)
                robot.update_tile(start_position, sensor_data, current_direction, robot.start_point, L_v, V_v=0, processed_tiles = 0)
        
    def forward(self):
        self.leftMotor.setVelocity(self.MAX_SPEED)
        self.rightMotor.setVelocity(self.MAX_SPEED)
        
    def slow_forward(self):
        self.leftMotor.setVelocity(1.256)
        self.rightMotor.setVelocity(1.256)   
        
    def backward(self):
        self.leftMotor.setVelocity(-self.MAX_SPEED)
        self.rightMotor.setVelocity(-self.MAX_SPEED)
        
    def slow_backward(self):
        self.leftMotor.setVelocity(-1.256)
        self.rightMotor.setVelocity(-1.256)  
        
    def left(self):
        self.leftMotor.setVelocity(-1.57)
        self.rightMotor.setVelocity(1.57)
        
    def right(self):
        self.leftMotor.setVelocity(1.57)
        self.rightMotor.setVelocity(-1.57)
        
    def slow_left(self):
        self.leftMotor.setVelocity(-1.256)
        self.rightMotor.setVelocity(1.256)

    def slow_right(self):
        self.leftMotor.setVelocity(1.256)
        self.rightMotor.setVelocity(-1.256)
  
    def stop(self):
        self.leftMotor.setVelocity(0)
        self.rightMotor.setVelocity(0)    
        
    def rotate_to(self, target_yaw):
        """ Rotates the robot to one specific direction. """
        completed = False
        speed = 0.3
        # Are we rotating left or right?
        starting_yaw = self.get_yaw()
        print(f"🔄 Rotating to {target_yaw}° from {starting_yaw}°")
        # Calculate the difference between target and current angles
        angle_difference = target_yaw - starting_yaw
        # Ensure the angle difference is within the range [-180, 180]
        if angle_difference < -180:
            angle_difference += 360
        if angle_difference > 180:
            angle_difference -= 360
        # Determine the turn direction
        rotation_left = True if angle_difference > 0 else False
        
        while self.step(self.timestep) != -1:
            current_yaw = self.get_yaw()
            if abs(target_yaw - current_yaw) > self.ANGLE_THRESHOLD:
                if rotation_left:
                    leftSpeed = -speed * self.MAX_SPEED
                    rightSpeed = speed * self.MAX_SPEED
                else:
                    leftSpeed = speed * self.MAX_SPEED
                    rightSpeed = -speed * self.MAX_SPEED
            else:
                leftSpeed = 0.0
                rightSpeed = 0.0
                completed = True
            self.leftMotor.setVelocity(leftSpeed)
            self.rightMotor.setVelocity(rightSpeed)
            if completed:
                self.current_angle = target_yaw
                print(f"✅ Rotation complete. Current yaw: {self.get_yaw()}°")
                self.wait()
                return
    

    
    def turn_east(self):
        if self.verbose:
            print("Rotating EAST")
        self.rotate_to(0)
        
    def turn_north(self):
        if self.verbose:
            print("Rotating NORTH")
        self.rotate_to(90)
        
    def turn_west(self):
        if self.verbose:
            print("Rotating WEST")
        self.rotate_to(180)
        
    def turn_south(self):
        if self.verbose:
            print("Rotating SOUTH")
        self.rotate_to(270)
     
    def has_passed_target(self, start, target, current):
        """ Check if the robot has passed the target by comparing movement direction. """
        if self.current_angle == 0:  # EAST (+Z)
            return (current[2] - start[2]) > (target[2] - start[2])  # Passed if beyond target
        elif self.current_angle == 90:  # NORTH (-X)
            return (current[0] - start[0]) < (target[0] - start[0])  
        elif self.current_angle == 180:  # WEST (-Z)
            return (current[2] - start[2]) < (target[2] - start[2])  
        elif self.current_angle == 270:  # SOUTH (+X)
            return (current[0] - start[0]) > (target[0] - start[0])  
        return False
      
    def is_target_cell(self):
        """ Check if the current cell is the target cell. """
        current_position = get_robot_position()
        print(f"current_position : {current_position}")
        print(f"target_cell : {self.target_cell}")
        return hasattr(self, 'target_cell') and self.target_cell == current_position
    
    def move_forward(self):
        if self.verbose:
            print("🚀 Moving forward")

        self.leftMotor.setVelocity(0.5 * self.MAX_SPEED)
        self.rightMotor.setVelocity(0.5 * self.MAX_SPEED)

        # Initialize position tracking
        distance_traveled = 0
        self.prev_left_pos = self.left_position_sensor.getValue()
        self.prev_right_pos = self.right_position_sensor.getValue()

        completed = False

        while self.step(self.timestep) != -1:
            # Read new wheel values
            left_pos = self.left_position_sensor.getValue()
            right_pos = self.right_position_sensor.getValue()

            # Compute deltas
            delta_left = left_pos - self.prev_left_pos
            delta_right = right_pos - self.prev_right_pos

            # Update previous values
            self.prev_left_pos = left_pos
            self.prev_right_pos = right_pos

            # Compute distance traveled (average of both wheels)
            left_distance = delta_left * self.WHEEL_RADIUS
            right_distance = delta_right * self.WHEEL_RADIUS
            distance_traveled += (left_distance + right_distance) / 2.0

            #print(f"📏 Distance Traveled: {round(distance_traveled, 4)} m")

            # ✅ Stop when near center (tuned for your cell size)
            if 0.12 < round(distance_traveled, 4) < 0.13:
                print("✅ Arrived at cell center")
                completed = True

            if completed:
                self.leftMotor.setVelocity(0)
                self.rightMotor.setVelocity(0)
                self.wait()
                return
                        
  
    def get_robot_direction(self):
        """Returns the current direction the robot is facing (NORTH, EAST, SOUTH, WEST) based on the yaw value."""
        yaw = self.get_yaw()
    
        if 350 <= yaw or yaw < 10:  # 0° ± 10°
            return "EAST"
        elif 80 <= yaw <= 100:  # 90° ± 10°
            return "NORTH"
        elif 170 <= yaw <= 190:  # 180° ± 10°
            return "WEST"
        elif 260 <= yaw <= 280:  # 270° ± 10°
            return "SOUTH"
        else:
            return f"UNKNOWN (Yaw: {yaw}°)"    
                
                
robot = MyRobotMovement()
robot.initialize_devices()

gps = robot.gps  # GPS for position tracking
# N  S  E  W  V  H  C  VIC ST HS L
# 1  0  0  0  0  0  0  1

# Temporary static map (from user)
temporary_map = np.array([
    [0b1010000010 , 0b0010000000 , 0b0010000000 , 0b0110010000 , 0b1110000000 , 0b1010000000 , 0b0010000000 , 0b0110010000 ],
    [0b1000000000 , 0b0101000000 , 0b1000000000 , 0b1001000000 , 0b1000000000 , 0b0100000000 , 0b1001000000 , 0b0100000100 ],
    [0b1000000100 , 0b0110000000 , 0b1000000000 , 0b0011000000 , 0b0001000000 , 0b0100000000 , 0b1010000000 , 0b0100000000 ],
    [0b1000000000 , 0b0100000000 , 0b1000000000 , 0b0110000000 , 0b1010000000 , 0b0101000000 , 0b1100000000 , 0b1101000000 ],
    [0b1000000000 , 0b0000000000 , 0b0000000000 , 0b0000000000 , 0b0110000000 , 0b1010000000 , 0b0000000000 , 0b0110000000 ],
    [0b1000000000 , 0b0100000000 , 0b1101000100 , 0b1000000000 , 0b0100000000 , 0b1000000100 , 0b1111000000 , 0b0100000000 ],
    [0b1000000100 , 0b0000000000 , 0b0110000000 , 0b1000000000 , 0b0100000000 , 0b1000000000 , 0b1111000000 , 0b0100000000 ],
    [0b1001000000 , 0b0001000000 , 0b0001000000 , 0b0001000000 , 0b0101001000 , 0b0001000000 , 0b0001000000 , 0b0101000000 ]
])  

# ======================
# DATA STORAGE & RESET
# ======================

# Function to save processed sign classes and tile types to JSON file
def save_processed_data_to_json():
    with open("processed_data.json", "w") as f:
        json.dump({
            "processed_sign_classes": {str(k): v for k, v in processed_sign_classes.items()},
            "processed_tiles": {str(k): v for k, v in processed_tiles.items()}
        }, f, indent=2)
    print("✅ Processed data saved to processed_data.json")
    print(processed_sign_classes)

# Function to load processed sign classes and tiles from JSON file at startup
def load_processed_data_from_json():
    global processed_signs, processed_tiles, processed_sign_classes

    if os.path.exists("processed_data.json"):
        with open("processed_data.json", "r") as f:
            data = json.load(f)
            processed_sign_classes = {eval(k): v for k, v in data.get("processed_sign_classes", {}).items()}
            processed_tiles = {eval(k): v for k, v in data.get("processed_tiles", {}).items()}
            processed_signs = set(processed_sign_classes.keys())  # Rebuild processed_signs from loaded classes

        print("✅ Processed data loaded from processed_data.json")
    else:
        print("ℹ️ No saved data found. Starting fresh.")

# Function to reset the saved sign and tile data to empty at the beginning of each run
def reset_processed_data_to_json():
    global processed_signs, processed_tiles, processed_sign_classes

    processed_signs = set()
    processed_tiles = {}
    processed_sign_classes = {}

    with open("processed_data.json", "w") as f:
        json.dump({
            "processed_sign_classes": {},
            "processed_tiles": {}
        }, f, indent=2)

    print("🔄 Processed data reset at startup.")
                   
# ======================
# TEXT PROCESSING
# ======================
  
# Function to clean and normalize extracted text for classification
def preprocess_text(text):
    """Clean and normalize extracted text."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z0-9 ]', '', text)  # Remove special characters
    return text.strip()  # Remove leading/trailing whitespace
    
# Function to Extract Text from Image Using OCR
def extract_text_from_image(image_array):

    # Perform OCR on the input color image
    results = reader.readtext(image_array)

    # Check if OCR found any text
    if not results:
        print("🛑 No text detected by OCR.")
        return "No Text Detected"

    # Display OCR results for debugging
    print("📋 OCR Detection Results:")
    for (bbox, text, conf) in results:
        print(f"🧐 Text: '{text}' | Confidence: {conf:.2f}")

    # Combine all detected text into a single string
    extracted_text = " ".join([res[1] for res in results])
    return extracted_text

# Function to draw OCR bounding boxes and show detected text with confidence on the image
def show_ocr_boxes(image_array):
    """Displays OCR bounding boxes and detected text with confidence on the image."""
    
    # Run OCR to get list of (bounding box, text, confidence)
    results = reader.readtext(image_array)
    h, w = image_array.shape[:2]

    for (bbox, text, conf) in results:
        # Convert bounding box coordinates to integer type
        pts = np.array(bbox, dtype=np.int32)

        # Draw a green bounding box around detected text
        cv2.polylines(image_array, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # Extract the top-left point of the bounding box
        x, y = pts[0]

        # Clamp x to stay within image width (leave room for label text)
        x = max(0, min(x, w - 150))

        # Clamp y to stay within image height (leave room above the box)
        y = max(20, min(y, h - 10))

        # Create label text with confidence value
        label = f"{text} ({conf:.2f})"

        # Put the label text above the bounding box in blue
        cv2.putText(image_array, label, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
    # ✅ Show the image in a window
    #cv2.imshow("OCR Boxes", image_array)
    #cv2.waitKey(1)                  

                   
# ================================
# SIGN DETECTION AND HANDLING
# ================================

# Function to determine if a sign is red or white based on HSV color detection
def determine_sign_color(roi):
    # Convert image to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define red color ranges
    red_lower1, red_upper1 = (0, 50, 50), (10, 255, 255)
    red_lower2, red_upper2 = (170, 50, 50), (180, 255, 255)
    
    # Create red masks and combine
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Calculate red pixel ratio
    red_pixel_ratio = np.sum(red_mask) / (red_mask.size * 255)

    # Return color based on threshold
    if red_pixel_ratio > 0.05:
        return "red"
    return "white"

# Function to detect hazard signs (red, white, yellow) using color masks and rotated bounding boxes
def detect_sign(frame):
    """Detects hazard signs (red, white, yellow) using rotated bounding boxes with debug view."""
    global detected_sign, sign_classified

    # Crop top portion to reduce glare effect
    height, width = frame.shape[:2]
    crop_top = int(height * 0.1)
    frame_cropped = frame[crop_top:, :]

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2HSV)

    # Define color ranges for red, white, and yellow
    red_lower1, red_upper1 = (0, 50, 50), (10, 255, 255)
    red_lower2, red_upper2 = (170, 50, 50), (180, 255, 255)
    white_lower, white_upper = (0, 0, 200), (180, 50, 255)
    yellow_lower, yellow_upper = (20, 100, 100), (30, 255, 255)

    # Create combined color mask
    red_mask = cv2.bitwise_or(cv2.inRange(hsv, red_lower1, red_upper1),
                              cv2.inRange(hsv, red_lower2, red_upper2))
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    combined_mask = cv2.bitwise_or(red_mask, white_mask)
    combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)

    # Morphological operations to clean the mask
    kernel = np.ones((7, 7), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=1)

    debug_frame = frame.copy()

    # Find contours from the final mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if area < 1000:
            continue

        # Get rotated bounding box and shift to original coordinates
        rot_rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rot_rect)
        box = np.intp(box)
        box_shifted = box + [0, crop_top]

        # Calculate aspect ratio
        (center_x, center_y), (w, h), angle = rot_rect
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0

        # Draw detection info for debugging
        cv2.drawContours(debug_frame, [box_shifted], 0, (255, 0, 255), 2)
        top_left = tuple(box_shifted[1])
        cv2.putText(debug_frame, f"A:{int(area)}", (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(debug_frame, f"AR:{aspect_ratio:.2f}", (top_left[0], top_left[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Detection logic for normal-sized signs
        if MIN_AREA <= area <= MAX_AREA and 0.6 <= aspect_ratio <= 1.8:
            robot_x, robot_y = get_robot_position()
            if (robot_x, robot_y) in processed_signs:
                print(f"⚠️ Sign at ({robot_x}, {robot_y}) already processed. Ignoring...")
                continue

            x, y, w_, h_ = cv2.boundingRect(box)
            y += crop_top
            detected_sign = (x, y, w_, h_)
            sign_classified = False
            return True

        # Detection logic for close-up signs (too large)
        elif area > MAX_AREA and len(approx) >= 4 and 0.6 <= aspect_ratio <= 1.8:
            robot_x, robot_y = get_robot_position()
            if (robot_x, robot_y) in processed_signs:
                print(f"⚠️ Close sign at ({robot_x}, {robot_y}) already processed. Ignoring...")
                continue

            print(f"✅ Close Sign Detected with area {int(area)}")
            x, y, w_, h_ = cv2.boundingRect(box)
            y += crop_top
            detected_sign = (x, y, w_, h_)
            sign_classified = False
            return True

    return False

# Function to move the robot step-by-step toward a detected sign using camera feedback
def navigate_to_sign():
    """Navigates step-by-step to a detected sign using camera guidance."""
    global detected_sign, sign_classified

    if detected_sign is None:
        print("⚠️ No detected sign to navigate to. Skipping...")
        return

    print(f"📍 Navigating to Sign using A*-like camera steps...")

    while robot.step(TIME_STEP) != -1:
        # Capture current camera frame
        image = robot.camera.getImage()
        width = robot.camera.getWidth()
        height = robot.camera.getHeight()
        frame = np.frombuffer(image, np.uint8).reshape((height, width, 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Update sign detection dynamically
        if not detect_sign(frame):
            print("⚠️ Lost sight of sign. Returning to normal navigation...")
            return

        # Extract position and size of detected sign
        x, y, w, h = detected_sign
        sign_x = x + w // 2
        sign_area = w * h

        print(f"📌 Sign Center X={sign_x}, Area={sign_area}")

        # Stop and classify if close enough to the sign
        if sign_area >= STOP_DISTANCE:
            print("🛑 Close to sign. Classifying...")
            robot.stop()
            for _ in range(5):
                robot.step(TIME_STEP)

            if not sign_classified:
                classify_and_act()
                processed_signs.add(detected_sign)

            # Rotate to avoid re-detecting the same sign
            print("🔄 Rotating to avoid re-detection...")
            robot.leftMotor.setVelocity(-TURN_SPEED)
            robot.rightMotor.setVelocity(TURN_SPEED)
            for _ in range(10):
                robot.step(TIME_STEP)

            robot.stop()
            return

        # Decide movement direction based on sign's horizontal position
        current_cell = get_robot_position()
        next_cell = (current_cell[0] + 1, current_cell[1])  # Default: forward

        if sign_x < width // 3:
            print("↩️ Turning LEFT toward sign...")
            robot.rotate_to(90)
            next_cell = (current_cell[0], current_cell[1] - 1)
        elif sign_x > 2 * width // 3:
            print("↪️ Turning RIGHT toward sign...")
            robot.rotate_to(270)
            next_cell = (current_cell[0], current_cell[1] + 1)

        print(f"🚶‍♂️ Step toward: {next_cell}")
        move_to_target(next_cell)
      
# Function to classify a hazard sign using OCR and a trained text classification model
def classify_sign(image_array):
    """Extracts and cleans text, then classifies using the trained pipeline."""
    
    # Extract text from the image using OCR
    extracted_text = extract_text_from_image(image_array)
 
    # If no text is detected, skip classification
    if extracted_text == "No Text Detected":
        print("⚠ No text found on the detected sign. Skipping classification.")
        return "Unknown"
 
    # Preprocess the extracted text
    clean_text = preprocess_text(extracted_text)
 
    # If the cleaned text is empty, skip classification
    if not clean_text:
        print("⚠ Extracted text after cleaning is empty. Skipping.")
        return "Unknown"
 
    # Display OCR boxes for visual debugging
    show_ocr_boxes(image_array)
 
    # Predict the hazard type using the trained model pipeline
    predicted_class = pipeline.predict([clean_text])[0]
 
    # Log the result for debugging
    print(f"🔍 Raw Text: '{extracted_text}' | 🧹 Cleaned: '{clean_text}' | 🚨 Predicted: {predicted_class}")
    return predicted_class
    
# Function to classify the detected sign using OCR and take appropriate action
def classify_and_act():
    global detected_sign, sign_classified

    # Capture image from camera
    image = robot.camera.getImage()
    width = robot.camera.getWidth()
    height = robot.camera.getHeight()

    frame = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Extract bounding box area around the detected sign
    x, y, w, h = detected_sign
    MARGIN = 10
    y1, y2 = max(0, y - MARGIN), min(frame.shape[0], y + h + MARGIN)
    x1, x2 = max(0, x - MARGIN), min(frame.shape[1], x + w + MARGIN)

    sign_roi = frame[y1:y2, x1:x2]

    # Resize for consistent OCR input
    ocr_image = cv2.resize(sign_roi, (128, 128), interpolation=cv2.INTER_AREA)

    # Optional: display cropped image
    #cv2.imshow("Cropped Sign", ocr_image)
    #cv2.waitKey(1)

    # Classify sign using OCR + trained model
    predicted_sign = classify_sign(ocr_image)

    print(f"🚨 Detected Hazard Sign: {predicted_sign}")

    if predicted_sign == "Unknown":
        print("⚠ Skipping sign, no valid text found. Continuing movement...")
        detected_sign = None
        sign_classified = False
        return

    sign_classified = True

    # Store sign data
    robot_grid_x, robot_grid_y = get_robot_position()
    grid_pos = (robot_grid_x, robot_grid_y)

    processed_signs.add(grid_pos)  # Position set
    robot.processed_signs_dict[grid_pos] = predicted_sign
    processed_sign_classes[grid_pos] = predicted_sign  # Class mapping

    print(f"📌 Added {grid_pos} as {predicted_sign} to processed_signs.")
    print("🔄 Resuming A* Navigation to Target Cell...")
    
    save_processed_data_to_json()
    
        
# Function to estimate how many tiles ahead the sign is based on its detected area
def estimate_sign_tile_offset(sign_area):
    """
    Estimate how far the sign is (in number of tiles ahead) based on its bounding box area.
    You can calibrate these thresholds using real test data.
    """
    if sign_area > 24000:
        return 1  # Very close
    elif sign_area > 12000:
        return 2
    elif sign_area > 8000:
        return 3
    else:
        return 4  # Far 
        
# ======================
# TILE DETECTION
# ======================
                                                           
# Function to classify tile color and return its name and assigned number
def classify_tile(color):
    detected_r, detected_g, detected_b = color

    for tile_name, (r, g, b, number) in TILE_MAPPING.items():
        # Check if detected color is within tolerance range
        if (
            abs(detected_r - r) <= COLOR_TOLERANCE and
            abs(detected_g - g) <= COLOR_TOLERANCE and
            abs(detected_b - b) <= COLOR_TOLERANCE
        ):
            return tile_name, number  # Return the matched tile and its number

    return "Unknown", None  # No matching tile found
    
# Function to detect the current tile and store its type with the robot's grid position
def detect_and_store_tile():
    """Detects the tile color and stores it with the robot’s grid position."""
    global processed_tiles

    # Get the robot’s current grid position
    current_grid = get_robot_position()

    # Skip if this tile has already been recorded
    if current_grid in processed_tiles:
        return  

    # Read from the color sensor
    image_data = robot.color_sensor.getImage()
    if image_data:
        rgba_values = np.frombuffer(image_data, dtype=np.uint8)
        detected_color = (int(rgba_values[0]), int(rgba_values[1]), int(rgba_values[2]))

        # Classify tile type and number
        tile_type, tile_number = classify_tile(detected_color)

        # Store tile type if recognized
        if tile_type != "Unknown":
            processed_tiles[current_grid] = {
                "type": tile_type,
                "number": tile_number
            }
            robot.processed_tile_dict = processed_tiles
            print(f"🟨 Tile Stored: {tile_type} (#{tile_number}) at {current_grid}")
            
            save_processed_data_to_json()
                       
def get_robot_position():
    current_gps = gps.getValues()
    return get_grid_position_from_gps(current_gps, robot.gps_grid)

def get_grid_position_from_gps(current_gps, gps_grid):
    robot_x, _, robot_z = current_gps
    min_distance = float('inf')
    best_cell = (0, 0)

    for row in range(len(gps_grid)):
        for col in range(len(gps_grid[0])):
            cell_x, _, cell_z = gps_grid[row][col]
            dx = robot_x - cell_x
            dz = robot_z - cell_z
            distance = dx**2 + dz**2
            if distance < min_distance:
                min_distance = distance
                best_cell = (row, col)

    return best_cell
    

def heuristic(a, b):
    """Manhattan distance heuristic for A*."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_neighbors(position):
    """Get valid neighbors while considering walls and avoiding hazards."""
    x, y = position
    temporary_map2 = np.array(robot.grid)    
    # Simplified 3-value version (for get_neighbors):
    directions = [
        (-1, 0, 0b00100000000),  # UP    → wall on UP (E)
        (1, 0,  0b00010000000),  # DOWN  → wall on DOWN (W)
        (0, 1,  0b01000000000),  # RIGHT → wall on RIGHT (S) ✅
        (0, -1, 0b10000000000),  # LEFT  → wall on LEFT (N)
    ]

    neighbors = []

    for dx, dy, wall_mask in directions:
        new_x, new_y = x + dx, y + dy

        # ✅ Bounds check
        if 0 <= new_x < MAP_SIZE and 0 <= new_y < MAP_SIZE:
            current_cell = temporary_map2[x, y]
            neighbor_cell = temporary_map2[new_x, new_y]

            #print(f"🧭 Checking ({new_x},{new_y}) = {bin(neighbor_cell)} from ({x},{y}) = {bin(current_cell)} | wall_mask: {bin(wall_mask)}")
            # ✅ Log current + neighbor values before condition
            #print(f"➡️ Current ({x},{y}) = {bin(temporary_map[x][y])}, "
                  #f"Neighbor ({new_x},{new_y}) = {bin(temporary_map[new_x][new_y])}, "
                  #f"Wall mask: {bin(wall_mask)}")

            if not (temporary_map2[new_x, new_y] & 0b00000100000) and not (temporary_map2[x, y] & wall_mask):
                #print(f"✅ Added neighbor: ({new_x}, {new_y})")
                neighbors.append((new_x, new_y))
            #else:
                #print(f"❌ Skipped: ({new_x}, {new_y}) due to hazard or wall")
    return neighbors

def a_star_search(start, goal):
    """A* pathfinding that returns only the next best step toward the goal."""
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    temporary_map2 = np.array(robot.grid)
    
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        for neighbor in get_neighbors(current):
            new_cost = cost_so_far[current] + 1
            wall_count = bin(temporary_map2[neighbor[0], neighbor[1]] & 0b11110000000).count("1")
            adjusted_cost = new_cost + wall_count * 2
            if heuristic(neighbor, goal) >= heuristic(current, goal):
                adjusted_cost += 2
            if neighbor not in cost_so_far or adjusted_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = adjusted_cost
                priority = adjusted_cost + heuristic(goal, neighbor)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    # Reconstruct path
    path = []
    step = current
    while step is not None:
        path.append(step)
        step = came_from.get(step)
    path.reverse()

    # Always return next step only
    if len(path) > 1:
        return path[1]  # The next best cell
    return None  # No valid path

def avoid_side_wall(yaw):
    if yaw == 270:
        ps5 = robot.distance_sensors(5)
        ps6 = robot.distance_sensors(6)
        ps7 = robot.distance_sensors(7)

        print(f"🧱 ps5 (Left): {ps5}")
        print(f"🧱 ps6 (Left-Front): {ps6}")
        print(f"🧱 ps7 (Front-Left): {ps7}")

        if ps7 < 0.22 or (ps6 < 0.08 and ps7 < 0.35):
            print("↪️ Wall detected near front-left. Nudging right...")

            for _ in range(2):  # Double nudge
                robot.right()
                robot.step(10)
                robot.stop()
                robot.wait()

            #robot.move_forward()
            return  # ✅ Only exit early if nudge and move_forward triggered
                                                        
def move_to_target(next_cell):
    """ Move the robot to a single target cell (x, y). """
    robot_pos = get_robot_position()
    temporary_map2 = np.array(robot.grid)
    # ✅ Mark the current cell as visited before moving
    current_x, current_y = robot_pos
    temporary_map2[current_x, current_y] |= 0b0000100000

    x, y = next_cell
    robot.target_cell = next_cell
    print(f"🚀 Moving to ({x}, {y}) from {robot_pos}")

    dx = x - robot_pos[0]
    dy = y - robot_pos[1]

    expected_yaw = None
    if dx == -1:
        print("⬆️ Moving up (EAST)")
        expected_yaw = 0
    elif dx == 1:
        print("⬇️ Moving down (WEST)")
        expected_yaw = 180
    elif dy == 1:
        print("➡️ Moving right (SOUTH)")
        expected_yaw = 270
    elif dy == -1:
        print("⬅️ Moving left (NORTH)")
        expected_yaw = 90

    if expected_yaw is not None:
        robot.rotate_to(expected_yaw)
        #avoid_side_wall(expected_yaw)  # ✅ Nudge to avoid wall before moving forward
        
    robot.step(TIME_STEP)
    robot.move_forward()
    robot.step(TIME_STEP)

    # ✅ Detect and store tile after moving
    detect_and_store_tile()                                        
                           

    new_pos = get_robot_position()
    print(f"🔍 Expected: ({x}, {y}), Now at: {new_pos}")

    if new_pos == (x, y):
        print(f"✅ Reached ({x}, {y})")
        temporary_map2[x, y] |= 0b00001000000  # ✅ Mark the destination as visited
        
        # ✅ Convert LIDAR after arriving at the cell
        origin = robot.gps.getValues()
        current_direction = robot.get_direction()
        roll1, pitch1, yaw1 = robot.iu.getRollPitchYaw()
        combined_gps_points, _, _ = robot.get_lidar_points(origin, current_direction, yaw1)
        shared_wall_grid = robot.generate_shared_wall_grid(robot.gps_grid)
        robot.convert_lidar_points(combined_gps_points, robot.gps_grid, shared_wall_grid, current_direction)
        sensor_data = robot.get_sensor_data(current_direction)
        
        # Update the occupancy grid
        robot.update_tile(new_pos, sensor_data[0], origin, robot.start_point, L_v = 0, V_v = 1, processed_tiles = 0)
        robot.update_dict(new_pos, sensor_data[0], current_direction, origin)

        if robot.runonce_2 == 0:
            robot.grid[2][6] = 0b10100000001
            robot.runonce_2 = 1

        for row in robot.grid:
            print([format(cell, '011b') for cell in row])

        # Get the lidar points
        combined_gps_points, gps_x_point_1, gps_y_point_1 = robot.get_lidar_points(origin, current_direction, yaw1)
        cell_grid = robot.generate_shared_wall_grid(robot.gps_grid)

        # Plot the robot position
        robot.plot_position()

        # ✅ SCAN FOR SIGN after reaching the cell
        print("🔎 Scanning for signs after reaching the cell...")
        # Scan only in current direction
        yaw = robot.get_yaw()
        robot.rotate_to(yaw)  # Redundant but ensures alignment
        image = robot.camera.getImage()
        width = robot.camera.getWidth()
        height = robot.camera.getHeight()
        frame = np.frombuffer(image, np.uint8).reshape((height, width, 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        if detect_sign(frame):
            print("🚨 Sign Detected After Arriving at Cell!")
            navigate_to_sign()            
    else:
        print(f"⚠️ Retry move: Still at {new_pos}, expected {x, y}")
        robot.move_forward()
        robot.step(TIME_STEP)

# ======================
# MAIN LOOP 
# ======================

def detect_and_navigate():
    static_targets = [(2,0),(7, 0),(7,4), (5, 2), (2, 3), (2, 7), (7, 5), (7, 7)]
    target_index = 0
    
    prev_left_distance = 0
    prev_right_distance = 0
    prev_distance = prev_left_distance + prev_right_distance
    robot.prev_left_pos = robot.left_position_sensor.getValue()
    robot.prev_right_pos = robot.right_position_sensor.getValue()
    distance_traveled = robot.prev_left_pos + robot.prev_right_pos
    
    origin = robot.gps.getValues()
    initial_gps_val = [-0.48, -0.0118, -0.48]
    robot.gps_grid = robot.generate_gps_grid(initial_gps_val, 'topleft')
    current = get_robot_position()
    current_direction = robot.get_direction()
    roll1, pitch1, yaw1 = robot.iu.getRollPitchYaw()
    sensor_data, front, left, right = robot.get_sensor_data(current_direction)
    robot.update_dict((0,0), sensor_data, current_direction, origin)

    # get the lidar points and update the occupancy grid
    cell_grid = robot.generate_shared_wall_grid(robot.gps_grid)
    combined_gps_points, gps_x_point_1, gps_y_point_1 = robot.get_lidar_points(origin, current_direction, yaw1)
    robot.convert_lidar_points(combined_gps_points, robot.gps_grid, cell_grid, current_direction)
    for row in robot.grid:
        print([format(cell, '011b') for cell in row])

    # Plot the robot position
    robot.plot_position()

    while target_index < len(static_targets) and robot.step(TIME_STEP) != -1:
        target = static_targets[target_index]
        print(f"\n🎯 Target {target_index + 1}/{len(static_targets)}: {target}")

        while current != target and robot.step(TIME_STEP) != -1:
            print(f"📍 Current Position: {current}")
            next_cell = a_star_search(current, target)
            if not next_cell:
                print("⚠️ No path. Skipping to next target...")
                break

            print(f"➡️ Next Step: {next_cell}")
            move_to_target(next_cell)

            current = get_robot_position()
            

            # ✅ Get and process LIDAR points every step
            origin = robot.gps.getValues()
            current_direction = robot.get_direction()
            combined_gps_points, _, _ = robot.get_lidar_points(origin, current_direction, yaw1)
            # robot.convert_lidar_points(combined_gps_points, robot.gps_grid, cell_grid, current_direction)
            print(f"✅ Arrived at {current}")
            
        target_index += 1  # Move to the next target

    print("\n🎉 Finished navigation to all static targets.")
    save_processed_data_to_json()


if __name__ == "__main__":
    reset_processed_data_to_json()  # Always start with fresh signs/tiles
    detect_and_navigate()