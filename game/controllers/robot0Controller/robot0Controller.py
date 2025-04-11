import heapq
import math
import numpy as np
from controller import Robot
#from shape_detection_controller import Shape
import time
import cv2
import tensorflow as tf
import os
import random
import joblib
import easyocr
import sklearn

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Constants
TIME_STEP = 32
MAP_SIZE = 8  # Adjust as per actual map size
CELL_SIZE = 0.2 
CAMERA_NAME = "camera_centre"
MIN_AREA = 5000
MAX_AREA = 30000
FORWARD_SPEED = 3.0
TURN_SPEED = 0.5
STOP_DISTANCE = 500
MODEL_INPUT_SIZE = (128, 128)
TURN_DURATION = 40  # Adjust for 90-degree turn

# Load the trained CNN model
model_path = "/Users/chel/Downloads/text_classifier.pkl"
pipeline = joblib.load(model_path)

# ✅ Initialize OCR
reader = easyocr.Reader(['en'])

# ✅ Class-to-Letter Mapping (Ensures Output is a Single-Letter Class)
class_to_letter = {
    "flammable": "F",
    "corrosive": "C",
    "H": "H",
    "organic": "O",
    "poison": "P",
    "S": "S",
    "U": "U"
}

# Predefined tile colors (R, G, B) with assigned numbers
TILE_MAPPING = {
    "black": (60, 60, 60, 2),
    "swamp": (142, 222, 245, 3),
    "purple": (251, 91, 193, 7),
    "blue": (255, 91, 91, 6),
    "grey": (124, 117, 115, 2),
    "green": (48, 255, 48, 9),
    "red": (91, 91, 255, 8)
}     
                                                 
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
        self.START_TILE = 11
        self.NORTH_WALL = 10
        self.EAST_WALL = 9
        self.SOUTH_WALL = 8
        self.WEST_WALL = 7
        self.VISITED = 6
        self.HAZ1 = 5
        self.HAZ2 = 4
        self.VICH = 3
        self.VICS = 2
        self.VICU = 1
        self.CHECKPOINT = 0
        self.grid_map = {}
        
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
        self.initial = 0b000000000000  # Binary representation (same as 0)
        
        self.grid = [[self.initial] * self.grid_size for _ in range(self.grid_size)]  # Fill the grid
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
        # **Log Initial Yaw**
        initial_yaw = self.get_yaw()
        print(f"🧭 Initial Robot Yaw: {initial_yaw}° ({self.get_robot_direction()})")
        # **Force alignment to EAST (0°) if necessary**
        if initial_yaw != 0:
            print("🔄 Rotating to EAST (0°) for consistent start position...")
            self.rotate_to(0)
            print(f"✅ Robot now facing: {self.get_robot_direction()}")        
    
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
            
    def generate_gps_grid(self, initial_gps, robotpos, grid_size=8, cell_size=0.1152):

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
        
    def update_tile(self, position, sensor_data, direction, coordinates, start_point):
        """
        sensor_data: [N, E, S, W] (1 = wall, 0 = open)
        """
        x = position[0]
        y = position[1]
        self.grid_map[(x, y)] = {
            "N": sensor_data[0],
            "E": sensor_data[1],
            "S": sensor_data[2],
            "W": sensor_data[3],
            "V": True,  # Mark the tile as visited
            "O": direction,
            "GPS": coordinates
        }

        is_north_wall = True if sensor_data[0] == 1 else False
        is_east_wall = True if sensor_data[1] == 1 else False
        is_south_wall = True if sensor_data[2] == 1 else False
        is_west_wall = True if sensor_data[3] == 1 else False
        
        if start_point == True:
            tile_status = 0b100000000000  # Initial state (optional)
        else:
            tile_status = 0b000000000000  # Initial state (optional)
        
        
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
        


    def move_forward_old(self):
        """ Moves the robot forward to the next cell and stops exactly at the cell center while handling wall collisions. """
        if self.verbose:
            print("🚀 Moving forward")
    
        speed = 0.5  # Default speed
        CELL_SIZE = 0.12  # ✅ Consistent cell size
        STOP_TOLERANCE = CELL_SIZE / 2  # ✅ Increased tolerance
    
        starting_grid = get_robot_position()  # Get the current cell (grid)
        print(f"📌 Starting Grid Position: {starting_grid}")
    
        extra_distance = 0  # Initialize extra movement adjustment
    
        # ✅ Compute the **destination grid cell**
        if self.current_angle == 0:  # Moving UP (North)
            destination_grid = (starting_grid[0] - 1, starting_grid[1])
        elif self.current_angle == 90:  # Moving LEFT (West)
            destination_grid = (starting_grid[0], starting_grid[1] - 1)
        elif self.current_angle == 180:  # Moving DOWN (South)
            destination_grid = (starting_grid[0] + 1, starting_grid[1])
        elif self.current_angle == 270:  # Moving RIGHT (East)
            destination_grid = (starting_grid[0], starting_grid[1] + 1)
        else:
            print("⚠️ Unknown movement direction!")
            return
    
        print(f"📏 Expected Destination Grid: {destination_grid}")
    
        # ✅ Compute the exact **center** of the destination cell using CELL_SIZE
        center_x = -0.480 + (destination_grid[1] * CELL_SIZE) + (CELL_SIZE / 2)
        center_z = -0.480 + (destination_grid[0] * CELL_SIZE) + (CELL_SIZE / 2)
    
        print(f"🎯 Expected Center Position: X={center_x}, Z={center_z}")
    
        completed = False
        while self.step(self.timestep) != -1:
            current_grid_x, current_grid_y = get_robot_position()  # Get updated cell
            current_x, _, current_z = self.gps.getValues()  # Get exact GPS position
            print(f"📍 Current Grid Position: ({current_grid_x}, {current_grid_y})")
            print(f"📍 Current GPS Position: X={current_x}, Z={current_z}")
    
            # ✅ Step 1: Check if the robot is near the correct grid
            at_correct_grid = (abs(current_x - center_x) < CELL_SIZE / 2 and abs(current_z - center_z) < CELL_SIZE / 2)
    
            # ✅ Step 2: Check if the robot is at the exact center of the cell (with increased tolerance)
            at_center = (abs(current_x - center_x) < STOP_TOLERANCE and abs(current_z - center_z) < STOP_TOLERANCE)
    
            print(f"📍 at_correct_grid: {at_correct_grid}")
            print(f"📍 at_center: {at_center} (Tolerance: {STOP_TOLERANCE})")
    
            # ✅ Slow down earlier to prevent overshooting
            distance_to_center_x = abs(current_x - center_x)
            distance_to_center_z = abs(current_z - center_z)
            if distance_to_center_x < STOP_TOLERANCE * 2 and distance_to_center_z < STOP_TOLERANCE * 2:
                speed = 0.2  # Reduce speed when close to center
    
            if at_correct_grid and at_center:
                print(f"✅ Arrived at ({destination_grid[0]}, {destination_grid[1]}) (At cell center)")
                completed = True
    
            if completed:
                self.stop()
                self.wait()
                return
            else:
                self.leftMotor.setVelocity(speed * self.MAX_SPEED)
                self.rightMotor.setVelocity(speed * self.MAX_SPEED)
    def is_target_cell(self):
        """ Check if the current cell is the target cell. """
        current_position = get_robot_position()
        print(f"current_position : {current_position}")
        print(f"target_cell : {self.target_cell}")
        return hasattr(self, 'target_cell') and self.target_cell == current_position
    
        
    def move_forward(self):
        """ Moves the robot forward for a set distance while continuously checking for signs. """
        if self.verbose:
            print("Moving forward")
        speed = 0.5
        starting_coordinate = self.gps.getValues()
        print(f"🚀 Starting Position: {starting_coordinate}")
        print(f"🔄 Current Yaw: {self.current_angle}°")

        # Calculate the desired ending coordinate
        destination_coordinate = [
            starting_coordinate[0] + CELL_SIZE * math.cos(math.radians(self.current_angle)),
            starting_coordinate[1],
            starting_coordinate[2] - CELL_SIZE * math.sin(math.radians(self.current_angle))
        ]

        print(f"📏 destination_coordinate: {destination_coordinate}")
        completed = False
        while self.step(self.timestep) != -1:
            current_coordinate = self.gps.getValues()
            print(f"🚀 Current Position: {current_coordinate}")
            
            detect_and_store_tile()                       

            # ✅ **Check for signs continuously while moving**
            image = self.camera.getImage()
            width = self.camera.getWidth()
            height = self.camera.getHeight()
            frame = np.frombuffer(image, np.uint8).reshape((height, width, 4))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
           
            if detect_sign(frame):  
            
                if detected_sign in processed_signs:
                    print("⚠️ Sign at this location already processed. Ignoring...")
                    continue  # Skip this detection and keep moving
                
                print("🚨 Sign Detected! Interrupting Movement...")
                self.leftMotor.setVelocity(0)  # Stop movement
                self.rightMotor.setVelocity(0)
                self.step(TIME_STEP)
                
                navigate_to_sign()  # **Handle sign navigation**
            
                print("🔄 Recalculating A* Path and Resuming Navigation...")
                return  # 
                
            # ✅ **Movement logic (unchanged)**
            if self.current_angle == 180:  # Moving along X-axis (down)
                distance_to_target_x = abs(current_coordinate[0] - destination_coordinate[0])
                distance_to_target_z = abs(current_coordinate[2] - destination_coordinate[2])
                threshold_x = self.DISTANCE_THRESHOLD
                threshold_z =  0.1167 #self.DISTANCE_THRESHOLD_Z
                wall_distance =self.distance_sensors(0)
                wall_distance2 =self.distance_sensors(7)
                wall_distance3 =self.distance_sensors(6)
                wall_distance4 =self.distance_sensors(1)
                wall_distance5 =self.distance_sensors(5)
                wall_distance6 =self.distance_sensors(2)
                print(f"✅ 180wall_distance : {wall_distance}")
                print(f"✅ 180wall_distance2 : {wall_distance2}")
                print(f"✅ 180wall_distance3 : {wall_distance3}")
                print(f"✅ 180wall_distance4 : {wall_distance4}")
                print(f"✅ 180wall_distance5 : {wall_distance5}")
                print(f"✅ 180wall_distance6 : {wall_distance6}")
                  
                if distance_to_target_x > threshold_x and distance_to_target_z > threshold_z:
                    print("✅ Arrived at target location at 180")
                    completed = True
                elif hasattr(self, 'target_cell') and self.is_target_cell():
                    if self.distance_sensors(0) <= 0.025 or self.distance_sensors(7) <= 0.022:
                        print("🚧 Wall too close in front while heading WEST")
                        completed = True
                        
            elif self.current_angle == 0:
                distance_to_target_x = abs(current_coordinate[0] - destination_coordinate[0])
                distance_to_target_z = abs(current_coordinate[2] - destination_coordinate[2])
                threshold_x = self.DISTANCE_THRESHOLD
                threshold_z = self.DISTANCE_THRESHOLD_Z
                #print(f"🚀 Distance to Target X: {distance_to_target_x}, Threshold: {threshold_x}")
                #print(f"🚀 Distance to Target Z: {distance_to_target_z}, Threshold: {threshold_z}")
                if distance_to_target_x > threshold_x and 0.087 <= distance_to_target_z <= 0.097:
                #if distance_to_target_x > threshold_x and 0.034 <= distance_to_target_z <= 0.044:
                    print("✅ Arrived at target location at 0")
                    completed = True
                elif hasattr(self, 'target_cell') and self.is_target_cell():
                     if self.distance_sensors(0) <= 0.025 or self.distance_sensors(7) <= 0.022:
                        print("🚧 Wall too close in front while heading ESAT")
                        completed = True
                        
            elif self.current_angle == 90:
                distance_to_target_x = abs(current_coordinate[0] - destination_coordinate[0])
                distance_to_target_z = abs(current_coordinate[2] - destination_coordinate[2])
                threshold_x = 0.126
                threshold_z = 0.21
                if distance_to_target_x >= threshold_x and distance_to_target_z <= threshold_z:
                    print("✅ Arrived at target location at 90")
                    completed = True
                elif hasattr(self, 'target_cell') and self.is_target_cell():
                    print(f"✅ self.distance_sensors(0) : {self.distance_sensors(0)}")
                    print(f"✅ self.distance_sensors(7) : {self.distance_sensors(7)}")
                    if self.distance_sensors(0) <= 0.026 or self.distance_sensors(7) <= 0.023:
                        print("🚧 Wall too close in front while heading NORTH")
                        completed = True
                    #wall_distance =self.distance_sensors(0)
                    #wall_distance2 =self.distance_sensors(7)
                    #wall_distance3 =self.distance_sensors(6)
                    #wall_distance4 =self.distance_sensors(1)
                    #wall_distance5 =self.distance_sensors(5)
                    #wall_distance6 =self.distance_sensors(2)
                    #print(f"✅ wall_distance : {wall_distance}")
                    #print(f"✅ wall_distance2 : {wall_distance2}")
                    #print(f"✅ wall_distance3 : {wall_distance3}")
                    #print(f"✅ wall_distance4 : {wall_distance4}")
                    #print(f"✅ wall_distance5 : {wall_distance5}")
                    #print(f"✅ wall_distance6 : {wall_distance6}")  
                    #if self.distance_sensors(0) < 0.012 or self.distance_sensors(7) < 0.012:
                    #if self.distance_sensors(0) <= 0.025 or self.distance_sensors(7) <= 0.022:
                        #print("🚧 Wall too close in front while heading NORTH")
                        
                        # Get current grid position
                        #current_x, current_y = get_robot_position()
                        
                        # Read wall flags from temporary_map
                        #current_tile = temporary_map[current_x][current_y]
                        #wall_E = not (current_tile & 0b00100000)  # E
                        #wall_W = not (current_tile & 0b00010000)  # W
                        #wall_S = not (current_tile & 0b01000000)  # S
                       
                        #wall_E = not (current_tile & 0b0000100000)  # E = UP
                        #wall_W = not (current_tile & 0b0000001000)  # W = DOWN
                        #wall_S = not (current_tile & 0b0001000000)  # S = RIGHT
                        #wall_N = not (current_tile & 0b1000000000)  # N = LEFT
                        # Try to slide right (EAST)
                        #if wall_E:  # E = UP
                            #print("⬆ Adjusting slightly upward (EAST)")
                            #self.forward()
                            #self.step(5)
                            #self.stop()
                        #elif wall_W:  # W = DOWN
                            #print("⬇ Adjusting slightly downward (WEST)")
                            #self.backward()
                            #self.step(5)
                            #self.stop()
                        #elif wall_S:  # S = RIGHT
                            #print("➡ Adjusting slightly right (SOUTH)")
                            #self.right()
                            #self.step(5)
                            #self.stop()
                        #elif wall_N:  # N = LEFT
                            #print("⬅ Adjusting slightly left (NORTH)")
                            #self.left()
                            #self.step(5)
                            #self.stop()
                    
                        #self.wait()
                        #completed = True                

            else:  # Moving along Y-axis (left/right)
                distance_to_target_x = abs(current_coordinate[0] - destination_coordinate[0])
                distance_to_target_z = abs(current_coordinate[2] - destination_coordinate[2])
                threshold_x = 0.126 #0.11
                threshold_z = 0.21
                if hasattr(self, 'target_cell') :
                    print("target_cell is passed")
                if self.is_target_cell():
                    print("is_target_cell is passed")
                
                wall_distance =self.distance_sensors(0)
                wall_distance2 =self.distance_sensors(7)
                wall_distance3 =self.distance_sensors(6)
                wall_distance4 =self.distance_sensors(1)
                wall_distance5 =self.distance_sensors(5)
                wall_distance6 =self.distance_sensors(2)
                print(f"✅ wall_distance : {wall_distance}")
                print(f"✅ wall_distance2 : {wall_distance2}")
                print(f"✅ wall_distance3 : {wall_distance3}")
                print(f"✅ wall_distance4 : {wall_distance4}")
                print(f"✅ wall_distance5 : {wall_distance5}")
                print(f"✅ wall_distance6 : {wall_distance6}")  
                #if  wall_distance6 <= 0.02:
                    #print("🚨 Too close to the wall, but at target. Forcing stop at 0")      
                    #self.right()
                    #self.step(5)
                    #self.stop()
                    #self.wait()
                    #continue  
                        
                if distance_to_target_x >= threshold_x and distance_to_target_z <= threshold_z:
                    print("✅ Arrived at target location at 270")
                    completed = True
                elif hasattr(self, 'target_cell') and self.is_target_cell():
                    target_row, target_col = self.target_cell
                    center_x, _, center_z = self.gps_grid[target_row][target_col]
                    robot_x, _, robot_z = self.gps.getValues()
                
                    distance_to_center = math.sqrt((robot_x - center_x)**2 + (robot_z - center_z)**2)
                    print(f"📍 Robot Position: ({robot_x:.4f}, {robot_z:.4f})")
                    print(f"🎯 Cell Center:     ({center_x:.4f}, {center_z:.4f})")
                    print(f"📏 Distance to Center: {distance_to_center:.5f}")
                    #if wall_distance <= 0.031 or wall_distance2 <= 0.019 :  # Adjusted margin
                    #if wall_distance2 <= 0.025 and distance_to_target_x >= 0.09:
                   
                    # 🚨 Only allow emergency stop near center if robot is past halfway through cell
                    if distance_to_center <= 0.040 and distance_to_target_x >= 0.08 and (wall_distance <= 0.031 or wall_distance2 <= 0.019):
                        print("🚨 Emergency stop: near center, past halfway, and wall is dangerously close")
                        completed = True
                    
                    # 🚨 Allow hard fail-safe if robot is far and stuck
                    elif distance_to_target_x >= 0.09 and (wall_distance <= 0.026 or wall_distance2 <= 0.017):
                        print("🚨 Fail-safe: robot passed 90% of cell and wall too close")
                        completed = True

                #elif  not self.is_target_cell(): 
                    #print(f"Arrived wall_distance6 = {wall_distance6}")
                    #if wall_distance6 <= 0.02 :  # Adjusted margin
                        #print("🚨 Too close to the wall, but at target. Forcing stop at 0")      
                        #self.right()
                        #self.step(5)
                        #self.stop()
                        #self.wait()
                        #continue
                        
            print(f"🚀 Distance to Target X: {distance_to_target_x}, Threshold: {threshold_x}")
            print(f"🚀 Distance to Target Z: {distance_to_target_z}, Threshold: {threshold_z}")

            # ✅ **Stop movement if at the target location**
            if completed:
                leftSpeed = 0
                rightSpeed = 0                                
            else:
                leftSpeed = speed * self.MAX_SPEED
                rightSpeed = speed * self.MAX_SPEED

            self.leftMotor.setVelocity(leftSpeed)
            self.rightMotor.setVelocity(rightSpeed)

            # ✅ **Stop and wait before returning if movement is completed**
            if completed:
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
# N  S  E  W  V  H  C  VIC
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

# ✅ Function to Extract Text from Image Using OCR
def extract_text_from_image(image_array):
    """Extracts text from a Webots camera image."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    extracted_text = " ".join([res[1] for res in results]) if results else "No Text Detected"
    return extracted_text


# ✅ Function to Classify Hazard Sign Using OCR + Text Model
def classify_sign(image_array):
    """Extracts text and predicts the hazard sign, but skips classification if no text is detected."""
    extracted_text = extract_text_from_image(image_array)

    # ✅ If No Text is Detected, Skip Classification
    if extracted_text == "No Text Detected":
        print("⚠ No text found on the detected sign. Skipping classification.")
        return "Unknown"  # Or return None if preferred

    predicted_class = pipeline.predict([extracted_text])[0]

    # ✅ Print Extracted Text + Detected Sign
    print(f"🔍 Extracted Text: '{extracted_text}' | 🚨 Detected Hazard Sign: {predicted_class}")

    return predicted_class
    

def determine_sign_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Red color range
    red_lower1, red_upper1 = (0, 50, 50), (10, 255, 255)
    red_lower2, red_upper2 = (170, 50, 50), (180, 255, 255)
    
    # Create red mask
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    red_pixel_ratio = np.sum(red_mask) / (red_mask.size * 255)

    if red_pixel_ratio > 0.05:  # Threshold to determine if a sign is red
        return "red"
    return "white"

# Function to detect signs
                          
def detect_sign(frame):
    """Detects hazard signs while ignoring the top 20% of the frame."""
    global detected_sign, sign_classified

    height, width = frame.shape[:2]

    # ✅ CROP: Ignore the top 20% of the frame (to remove sunlight issues)
    crop_top = int(height * 0.2)
                                    
    frame_cropped = frame[crop_top:, :]  # Remove the top portion

    hsv = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2HSV)

    # ✅ Define color ranges
    red_lower1, red_upper1 = (0, 50, 50), (10, 255, 255)
    red_lower2, red_upper2 = (170, 50, 50), (180, 255, 255)
    white_lower, white_upper = (0, 0, 200), (180, 50, 255)
    yellow_lower, yellow_upper = (20, 100, 100), (30, 255, 255)

    # ✅ Create masks
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    white_mask = cv2.inRange(hsv, white_lower, white_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)

    # ✅ Process each color mask
    for mask in [red_mask, white_mask, yellow_mask]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)

            if MIN_AREA <= area <= MAX_AREA:
                x, y, w, h = cv2.boundingRect(approx)

                # ✅ Shift y-coordinates back to match the original frame
                y += crop_top

                # ✅ Get the robot's current grid position
                robot_x, robot_y = get_robot_position()

                # ✅ Check if the detected sign's grid position is already processed
                if (robot_x, robot_y) in processed_signs:
                    print(f"⚠️ Sign at ({robot_x}, {robot_y}) already processed. Ignoring...")
                    return False  # Skip this sign

                # ✅ Store detected sign
                detected_sign = (x, y, w, h)
                sign_classified = False
                return True  # ✅ Return True immediately after detecting a valid sign
                                                  
                                 
                                                             
                                             
                                         

    return False  # ✅ If no sign is found, return False
                                                   

# Store signs that have already been classified
processed_signs = set() 

TOLERANCE_RADIUS = 1  # Allow some flexibility in position
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
        return 4  # Far away (likely max visible)
 
def navigate_to_sign():
    """Navigates to a detected sign using camera guidance, one grid at a time."""
    global detected_sign, sign_classified

    if detected_sign is None:
        print("⚠️ No detected sign to navigate to. Skipping...")
        return

    print(f"📍 Navigating to Sign (Using Camera Only)...")
                                           
    while robot.step(TIME_STEP) != -1:
        # ✅ Capture a new frame to update sign position dynamically
        image = robot.camera.getImage()
        width = robot.camera.getWidth()
        height = robot.camera.getHeight()
        frame = np.frombuffer(image, np.uint8).reshape((height, width, 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # ✅ Check if the sign is still visible
        if not detect_sign(frame):
            print("⚠️ Lost sight of sign! Returning to normal navigation...")
            return  # Exit if the sign disappears

        # ✅ Update sign properties dynamically
        x, y, w, h = detected_sign
        sign_x = x + w // 2  # Center X position of the detected sign
        sign_area = w * h  # Area of detected sign

        print(f"📌 Updated Sign Position: X={sign_x}, Area={sign_area}")

        # ✅ Stop if the robot is close to the sign
        if sign_area >= STOP_DISTANCE:  
            print(f"🛑 Close to the sign (Area: {sign_area}). Stopping for classification...")
            robot.leftMotor.setVelocity(0)
            robot.rightMotor.setVelocity(0)

            # ✅ Ensure the robot is completely still before classification
            for _ in range(5):
                robot.step(TIME_STEP)

            if not sign_classified:
                print("🔍 Classifying Sign...")
                classify_and_act()  # ✅ Process the sign
                processed_signs.add(detected_sign)  # ✅ Add to processed list

            print("✅ Sign Processing Complete. Adjusting orientation to avoid re-detection...")
                                                             
            # ✅ Rotate slightly to avoid re-detecting the same sign
            robot.leftMotor.setVelocity(-TURN_SPEED)
            robot.rightMotor.setVelocity(TURN_SPEED)
            for _ in range(10):  # Small turn duration
                robot.step(TIME_STEP)
            
            robot.leftMotor.setVelocity(0)
            robot.rightMotor.setVelocity(0)
                     
            print("🔄 Orientation adjusted. Resuming navigation...")
            return  # ✅ Exit function after handling the sign
                                              

        # ✅ Move One Grid Cell at a Time (Same as `move_to_target`)
        current_grid = get_robot_position()
        print(f"📍 Current Grid: {current_grid}")
                                                                        

        # ✅ Detect and store tile at the robot's position
        detect_and_store_tile()
                                
        # ✅ Determine movement direction based on sign position
        if sign_x < width // 3:
            print("🔄 Turning Left Towards Sign...")
            robot.rotate_to(90)  # Face left (West)
        elif sign_x > 2 * (width // 3):
            print("🔄 Turning Right Towards Sign...")
            robot.rotate_to(270)  # Face right (East)
        else:
            print("🚀 Moving Forward One Grid Cell...")
            move_to_target((current_grid[0] + 1, current_grid[1]))  # Move down  
     
            
#  Function to Classify the Detected Sign and Take Action
def classify_and_act():
    global detected_sign, sign_classified

    image = robot.camera.getImage()
    width = robot.camera.getWidth()
    height = robot.camera.getHeight()

    frame = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    x, y, w, h = detected_sign
    MARGIN = 10
    y1, y2 = max(0, y - MARGIN), min(frame.shape[0], y + h + MARGIN)
    x1, x2 = max(0, x - MARGIN), min(frame.shape[1], x + w + MARGIN)

    sign_roi = frame[y1:y2, x1:x2]  # ✅ Cropped ROI after detection

    cv2.imshow("Cropped Sign", sign_roi)
    cv2.waitKey(1)

    # ✅ Classify Sign Using OCR + Text-Based Model
    predicted_sign = classify_sign(sign_roi)

    # ✅ If No Text was Detected, Skip and Keep Moving
    if predicted_sign == "Unknown":
        print("⚠ Skipping sign, no valid text found. Continuing movement...")
        detected_sign = None  # ✅ Clear the detected sign so robot keeps looking
        sign_classified = False
                                             
        return

    print(f"🚨 Detected Hazard Sign: {predicted_sign}")

    sign_classified = True  # Prevent reclassification

    # ✅ Get the robot's grid position and store it in processed_signs
    robot_grid_x, robot_grid_y = get_robot_position()
    processed_signs.add((robot_grid_x, robot_grid_y))  # ✅ Store grid position, not bounding box

    print(f"📌 Added ({robot_grid_x}, {robot_grid_y}) to processed_signs.")
    print(f"🔄 Resuming A* Navigation to Target Cell...")

                                                                                        
    return  # **Exit function, letting navigate_to_sign() handle the next step**

# Tolerance for color variations
COLOR_TOLERANCE = 15                                                             
# Function to classify tile and return its assigned number
def classify_tile(color):
    detected_r, detected_g, detected_b = color

    for tile_name, (r, g, b, number) in TILE_MAPPING.items():
        if (
            abs(detected_r - r) <= COLOR_TOLERANCE and
            abs(detected_g - g) <= COLOR_TOLERANCE and
            abs(detected_b - b) <= COLOR_TOLERANCE
        ):
            return tile_name, number  # Return the matched tile and its number

    return "Unknown", None  # If no match is found

# Function to detect tiles and store them with grid positions
# Store processed tiles
processed_tiles = {}

def detect_and_store_tile():
    """Detects the tile color and stores it with the robot’s grid position."""
    global processed_tiles

    # ✅ Get the robot’s current grid position
    current_grid = get_robot_position()

    # ✅ Skip if tile at this grid has already been recorded
    if current_grid in processed_tiles:
        return  

    # ✅ Read color sensor and classify tile
    image_data = robot.color_sensor.getImage()
    if image_data:
        rgba_values = np.frombuffer(image_data, dtype=np.uint8)
        detected_color = (int(rgba_values[0]), int(rgba_values[1]), int(rgba_values[2]))

        tile_type, tile_number = classify_tile(detected_color)

        # ✅ Store tile type with its grid position
        if tile_type != "Unknown":
            processed_tiles[current_grid] = tile_type
            print(f"🟨 Tile Stored: {tile_type} at {current_grid}")
                        
# Function to turn the robot 90 degrees
def turn_90_degrees():
    turn_time = 0
    robot.left_motor.setVelocity(TURN_SPEED)
    robot.right_motor.setVelocity(-TURN_SPEED)

    while robot.step(TIME_STEP) != -1 and turn_time < TURN_DURATION:
        turn_time += 1  # Step counter for turn duration

    robot.left_motor.setVelocity(0)
    robot.right_motor.setVelocity(0)
    

def get_robot_position_1():
    """Convert GPS coordinates to grid position without computing center."""
    
    position = gps.getValues()  # Read GPS values
    raw_x, raw_y, raw_z = position  # Webots GPS format: (X, Y, Z)

    # ✅ Convert raw GPS coordinates to grid index
    #grid_x = int(round((raw_z + 0.480) / 0.11))  # Row index
    #grid_y = int(round((raw_x + 0.480) / 0.12))  # Column index
    grid_x = int(math.floor((raw_z + 0.480) / 0.11))
    grid_y = int(math.floor((raw_x + 0.480) / 0.12))
    # ✅ Clamp to avoid IndexError
    grid_x = max(0, min(MAP_SIZE - 1, grid_x))
    grid_y = max(0, min(MAP_SIZE - 1, grid_y))

    # 🔍 Debugging logs
    print(f"📡 RAW GPS Values: X={raw_x}, Y={raw_y}, Z={raw_z}")
    print(f"📌 Mapped Grid Position: ({grid_x}, {grid_y})")

    return grid_x, grid_y  # ✅ Only return (row, column)

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

def get_neighbors_new(position):
    """Get valid neighbors while considering walls in both directions and avoiding hazards."""
    x, y = position

    directions = [
        (0, 1, 0b0010000000, 0b0001000000),   # East
        (1, 0, 0b0100000000, 0b1000000000),   # South
        (0, -1, 0b0001000000, 0b0010000000),  # West
        (-1, 0, 0b1000000000, 0b0100000000)   # North
    ]

    neighbors = []

    for dx, dy, wall_mask, reverse_mask in directions:
        new_x, new_y = x + dx, y + dy

        if 0 <= new_x < MAP_SIZE and 0 <= new_y < MAP_SIZE:
            current = temporary_map[x, y]
            neighbor = temporary_map[new_x, new_y]

            print(f"🧭 Checking ({new_x},{new_y}) = {bin(neighbor)} from ({x},{y}) = {bin(current)}")

            reason = []
            if current & wall_mask:
                reason.append("wall in current")
            if neighbor & reverse_mask:
                reason.append("wall in neighbor")
            if neighbor & 0b0000010000:
                reason.append("hazard")

            if reason:
                print(f"❌ Skipped: ({new_x}, {new_y}) due to {' & '.join(reason)}")
            else:
                print(f"✅ Added neighbor: ({new_x}, {new_y})")
                neighbors.append((new_x, new_y))

    return neighbors



def get_neighbors(position):
    """Get valid neighbors while considering walls and avoiding hazards."""
    x, y = position
        
    # Simplified 3-value version (for get_neighbors):
    directions = [
        (-1, 0, 0b0010000000),  # East (UP)
        (0, 1, 0b0100000000),   # South (RIGHT)
        (1, 0, 0b0001000000),   # West (DOWN)
        (0, -1, 0b1000000000),  # North (LEFT)
    ]
    
    neighbors = []

    for dx, dy, wall_mask in directions:
        new_x, new_y = x + dx, y + dy

        # ✅ Bounds check
        if 0 <= new_x < MAP_SIZE and 0 <= new_y < MAP_SIZE:
            current_cell = temporary_map[x, y]
            neighbor_cell = temporary_map[new_x, new_y]

            #print(f"🧭 Checking ({new_x},{new_y}) = {bin(neighbor_cell)} from ({x},{y}) = {bin(current_cell)} | wall_mask: {bin(wall_mask)}")
            # ✅ Log current + neighbor values before condition
            #print(f"➡️ Current ({x},{y}) = {bin(temporary_map[x][y])}, "
                  #f"Neighbor ({new_x},{new_y}) = {bin(temporary_map[new_x][new_y])}, "
                  #f"Wall mask: {bin(wall_mask)}")

            if not (temporary_map[new_x, new_y] & 0b0000010000) and not (temporary_map[x, y] & wall_mask):
                #print(f"✅ Added neighbor: ({new_x}, {new_y})")
                neighbors.append((new_x, new_y))
            #else:
                #print(f"❌ Skipped: ({new_x}, {new_y}) due to hazard or wall")
    return neighbors


def get_neighbors_notallowvisited(position, backtrack=False):
    x, y = position
    directions = [(0, 1, 0b0010000000), (1, 0, 0b0100000000), (0, -1, 0b0001000000), (-1, 0, 0b1000000000)]
    
    unvisited_neighbors = []
    visited_neighbors = []

    for dx, dy, wall_mask in directions:
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < MAP_SIZE and 0 <= new_y < MAP_SIZE:
            is_wall_blocked = temporary_map[x, y] & wall_mask
            is_hazard = temporary_map[new_x, new_y] & 0b0000001000
            is_visited = temporary_map[new_x, new_y] & 0b0000100000

            if not is_wall_blocked and not is_hazard:
                if not is_visited:
                    unvisited_neighbors.append((new_x, new_y))
                else:
                    visited_neighbors.append((new_x, new_y))

    if unvisited_neighbors:
        return unvisited_neighbors
    elif backtrack and visited_neighbors:
        print(f"🔄 Backtracking: only visited neighbors available from {position}")
        return visited_neighbors
    else:
        print(f"🛑 No valid neighbors from {position}, returning self.")
        return [position]


def a_star_search_old(start, goal):
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        for neighbor in get_neighbors(current):
            new_cost = cost_so_far[current] + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + abs(goal[0]-neighbor[0]) + abs(goal[1]-neighbor[1])
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    path = []
    step = goal
    while step is not None:
        path.append(step)
        step = came_from.get(step)
    path.reverse()
    return path
    

def a_star_search(start, goal):
    """A* pathfinding that returns only the next best step toward the goal."""
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        for neighbor in get_neighbors(current):
            new_cost = cost_so_far[current] + 1
            wall_count = bin(temporary_map[neighbor[0], neighbor[1]] & 0b1111000000).count("1")
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



def has_reached_target(current_pos, target_pos, threshold=0.05):
    """
    Check if the robot has reached the target position within a small threshold 
    to avoid GPS floating-point errors.
    
    :param current_pos: (x, y) tuple of robot's current position
    :param target_pos: (x, y) tuple of target position
    :param threshold: Allowed margin of error to consider the target reached
    :return: True if within the threshold, False otherwise
    """
    return abs(current_pos[0] - target_pos[0]) < threshold and abs(current_pos[1] - target_pos[1]) < threshold

def avoid_side_wall_old(yaw):
    if yaw == 270:
        ps5 = robot.distance_sensors(5)
        ps6 = robot.distance_sensors(6)
        ps7 = robot.distance_sensors(7)

        print(f"🧱 ps5 (Left): {ps5}")
        print(f"🧱 ps6 (Left-Front): {ps6}")
        print(f"🧱 ps7 (Front-Left): {ps7}")

        if ps7 < 0.22 or (ps6 < 0.08 and ps7 < 0.35):
            print("↪️ Wall detected near front-left. Nudging right...")
            robot.right()
            robot.step(10)
            robot.stop()
            robot.wait()

            robot.move_forward()

            ps6_new = robot.distance_sensors(6)
            ps7_new = robot.distance_sensors(7)
            print(f"🔁 Re-check ps6: {ps6_new}, ps7: {ps7_new}")

            if ps7_new < 0.22 or (ps6_new < 0.08 and ps7_new < 0.35):
                print("⚠️ Still close — second nudge...")
                robot.right()
                robot.step(10)
                robot.stop()
                robot.wait()
        else:
            print("✅ Clear. No need to adjust.")


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
    
    # ✅ Mark the current cell as visited before moving
    current_x, current_y = robot_pos
    temporary_map[current_x, current_y] |= 0b0000100000

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
        avoid_side_wall(expected_yaw)  # ✅ Nudge to avoid wall before moving forward
        
    robot.step(TIME_STEP)
    robot.move_forward()
    robot.step(TIME_STEP)

    # ✅ Detect and store tile after moving
    detect_and_store_tile()                                        
                           

    new_pos = get_robot_position()
    print(f"🔍 Expected: ({x}, {y}), Now at: {new_pos}")

    if new_pos == (x, y):
        print(f"✅ Reached ({x}, {y})")
        temporary_map[x, y] |= 0b0000100000  # ✅ Mark the destination as visited
    else:
        print(f"⚠️ Retry move: Still at {new_pos}, expected {x, y}")
        robot.move_forward()
        robot.step(TIME_STEP)



def move_to_target_old(path):
    """ Move the robot along the computed path, ensuring correct alignment and position updates. """
    
    robot_pos = get_robot_position()  # ✅ Get the actual robot position before starting
    path_index = 0  # Track which part of the path the robot should follow
    # ✅ Mark start position as visited
    start_x, start_y = path[0]
    temporary_map[start_x, start_y] |= 0b00001000
    # ✅ If the robot is already at path[0], start from the next step
    if robot_pos == path[0]:
        print(f"⚠️ Robot is already at {path[0]}, skipping to next step...")
        path_index = 1  # Skip the first step if already at the starting point

    while path_index < len(path):
        x, y = path[path_index]
        robot.target_cell = path[path_index]
        print(f"🚀 Moving to ({x}, {y}) from {robot_pos}")

        dx = x - robot_pos[0]
        dy = y - robot_pos[1]

        expected_yaw = None
        if dx > 0:
            print("➡️ Moving down") 
            expected_yaw = 180
        elif dx < 0:
            print("⬅️ Moving up")
            expected_yaw = 0				
        elif dy > 0:
            print("🔽 Moving right")
            expected_yaw = 270
        elif dy < 0:
            print("🔼 Moving left")
            expected_yaw = 90

        if expected_yaw is not None:
            robot.rotate_to(expected_yaw)

        # **Ensure GPS updates before checking position**
        robot.step(TIME_STEP)
        robot.move_forward()
        robot.step(TIME_STEP)  # ✅ Ensure GPS updates after moving

        # **Update robot's current position**
        robot_pos = get_robot_position()
        print(f"🔍 Expected Position: ({x}, {y}), Current Position: {robot_pos}")

        # **Check if robot reached the expected position**
        if robot_pos == (x, y):
            print(f"✅ Successfully reached ({x}, {y})")
            path_index += 1  # ✅ Move to the next waypoint in the path
             # ✅ Mark visited bit
            temporary_map[x, y] |= 0b00001000
        else:
            print(f"⚠️ Position mismatch! Expected {x, y}, but robot at {robot_pos}. Retrying movement...")
            robot.move_forward()
            robot.step(TIME_STEP)
            robot_pos = get_robot_position()

        if robot_pos == path[-1]:  # **Final target reached**
            print("✅ Reached the victim at target location!")

            # **Mark the victim's location as visited**
            temporary_map[x, y] |= 0b00001000  # Set visited bit
            #print(f"📌 Updated tile ({x}, {y}): {temporary_map[x, y]:08b}")
            
            # **Find the next nearest victim**
            next_target = find_victims()
            if next_target:
                print(f"🔄 Replanning path to next victim at {next_target}...")
                new_path = a_star_search(robot_pos, next_target)
                if new_path:
                    move_to_target(new_path)  # **Re-run move_to_target() for next victim**
                else:
                    print("⚠️ No valid path found to next victim.")
            else:
                print("✅ All victims rescued. Proceeding to checkpoint.")
            return  # Exit current function after reaching the victim

    print("✅ All victims in this path have been rescued!")
    
    
def find_checkpoint():
    """Find checkpoint location."""
    for x in range(MAP_SIZE):
        for y in range(MAP_SIZE):
            if temporary_map[x, y] & 0b00000100:  # Checkpoint bit check
                return (x, y)
    return None
    
def find_victims():
    #print("🔍 Checking for all victims in the temporary map...")
    print("🗺️ Temporary Map Representation (Numeric):")
    for row in temporary_map:
        print([f'{cell:08b}' for cell in row])
    """Find the nearest victim based on the known map structure."""
    start = get_robot_position()
    victims = []
    for x in range(MAP_SIZE):
        for y in range(MAP_SIZE):
            cell_value = temporary_map[x, y]
            if (cell_value & 0b00000001) and not (cell_value & 0b00001000):   # Victim bit set
                walls = {
                    "N": bool(cell_value & 0b10000000),
                    "S": bool(cell_value & 0b01000000),
                    "E": bool(cell_value & 0b00100000),
                    "W": bool(cell_value & 0b00010000),
                }
                if all(walls.values()):  # 🚧 If all walls are True, victim is trapped
                    print(f"⚠️ Victim at ({x}, {y}) is completely trapped and unreachable!")
                else:
                    victims.append((x, y))
                    
    print(f"📝 Detected victims at: {victims}")  
 
    if not victims:
        return None
    
    victim_paths = [(victim, a_star_search(start, victim)) for victim in victims]
    victim_paths = [(v, p) for v, p in victim_paths if p]
    
    if not victim_paths:
        return None
    
    nearest_victim, _ = min(victim_paths, key=lambda v: len(v[1]))
    print(f"✅ Nearest victim based on pathfinding: {nearest_victim}")
    return nearest_victim
    
def choose_random_target():
    """Choose any random cell in 8x8 grid."""
    return random.choice([(x, y) for x in range(8) for y in range(8)])

def choose_random_target_visited():
    unvisited = [
        (x, y) for x in range(8) for y in range(8)
        if not (temporary_map[x, y] & 0b00001000)
    ]

    if unvisited:
        return random.choice(unvisited)
    
    # 🔁 If all cells visited or stuck, fallback to current position
    print("⚠️ All reachable cells visited or no options. Staying on current cell.")
    return get_robot_position()



def detect_and_navigate_1():
    """Main loop that integrates navigation and computer vision."""

    random_target = choose_random_target()
    print(f"🎯 Initial Random Target: {random_target}")

    while robot.step(TIME_STEP) != -1:
        print(f"🧭 Navigating to Target {random_target}...")
    
        # ✅ Pass both `robot` and `path`
        start = get_robot_position()
        path = a_star_search(start, random_target)
        move_to_target(path)  # 🚀 **Fix: Now passing the robot!**

        print(f"✅ Reached Target {random_target}. Selecting New Target...")
        random_target = choose_random_target()  # Pick a new target and continue

    print("✅ Mission Complete: Random Exploration Done!")
    
  
def detect_and_navigate_beforestatictarget():
    current = get_robot_position()
    target = choose_random_target()
    print(f"🎯 Random Target: {target}")

    while current != target and robot.step(TIME_STEP) != -1:
        print(f"📍 Current Position: {current}")

        next_cell = a_star_search(current, target)  # returns next cell only
        if not next_cell:
            print("⚠️ No path or stuck. Choosing new target...")
            target = choose_random_target()
            continue

        print(f"➡️ Next Step: {next_cell}")
        move_to_target(next_cell)  # pass just one cell

        current = get_robot_position()
        print(f"✅ Arrived at {current}")

    print("🎉 Finished navigation to final target.")

def detect_and_navigate():
    static_targets = [(5, 2), (7, 0), (7, 4), (2, 7), (7, 5), (7, 7)]
    target_index = 0
    origin = robot.gps.getValues()
    robot.gps_grid = robot.generate_gps_grid(origin, 'topleft')
    current = get_robot_position()

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
            print(f"✅ Arrived at {current}")

        target_index += 1  # Move to the next target

    print("\n🎉 Finished navigation to all static targets.")

def detect_and_navigate_new():
    # List of targets to visit
    static_targets = [(0, 0), (5, 2), (7, 0), (7, 4), (2, 7), (7, 5), (7, 7)]

    # Create full route: forward path + reversed path (excluding duplicate last point)
    full_route = static_targets + static_targets[::-1][1:]

    for target_index, target in enumerate(full_route):
        print(f"\n🎯 Target {target_index + 1}/{len(full_route)}: {target}")
        current = get_robot_position()

        # Keep navigating to the current target
        while current != target and robot.step(TIME_STEP) != -1:
            print(f"📍 Current Position: {current}")
            next_cell = a_star_search(current, target)

            if not next_cell:
                print("⚠️ No path. Skipping to next target...")
                break

            print(f"➡️ Next Step: {next_cell}")
            move_to_target(next_cell)

            current = get_robot_position()
            print(f"✅ Arrived at {current}")

    print("\n🎉 Finished navigation and returned to starting point.")

detect_and_navigate()