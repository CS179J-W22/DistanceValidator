import cv2
import imutils
from math import cos, sin, pi, floor, sqrt
from rplidar import RPLidar, RPLidarException
import numpy as np
from pynq import DefaultIP, Overlay
from IPython.display import clear_output
import ipywidgets as widgets 
import sys
import logging

# Constants
CAMERA_WIDTH = 640
LIDAR_ANGLE_MIN = 45
LIDAR_ANGLE_MAX = 135
SOCIAL_DISTANCE_THRESHOLD_MM = 1828.8  # 6 feet in millimeters
SCAN_SAMPLES_REQUIRED = 5
LIDAR_SCAN_POINTS = 360

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Note: LidarBoostDriver class and overlay are kept for potential future use
# but are no longer used in current arithmetic operations (native Python is faster)

class LidarBoostDriver(DefaultIP):
    def __init__(self, description):
        super().__init__(description=description)

    bindto = ['test.com:lidarBoost:lidarBoost:1.1']

    def add(self, a, b):
        self.write(0x10, a)
        self.write(0x18, b)
        self.write(0x30, 1)
        return self.read(0x20)
    
    def multiply(self, a, b):
        self.write(0x10, a)
        self.write(0x18, b)
        self.write(0x30, 0)
        return self.read(0x20)

# Overlay initialization - kept for compatibility but not actively used
overlay = Overlay('lidarBoost.bit')
haar_upper_body_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")
print_counter = 0

occupancy_out = widgets.Output()
with occupancy_out:
    display("loading...")
display(occupancy_out)

distance_out = widgets.Output()
with distance_out:
    display("loading...")
display(distance_out)

status_out = widgets.Output()
with status_out:
    display("loading...")
display(status_out)

scan_data = [0] * LIDAR_SCAN_POINTS

def map_x(x_val):
    """
    Maps x-coordinate from camera width to lidar angle range.
    
    Args:
        x_val: X-coordinate value from camera (0-640)
        
    Returns:
        Mapped angle value (45-135 degrees)
    """
    old_value = x_val
    old_min = 0
    old_max = CAMERA_WIDTH
    new_min = LIDAR_ANGLE_MIN
    new_max = LIDAR_ANGLE_MAX

    old_range = old_max - old_min

    if old_range == 0:
        return new_min
    
    new_range = new_max - new_min
    old_diff = old_value - old_min
    new_value = (old_diff * new_range / old_range) + new_min

    return int(new_value)

def get_position(scan_data, body_angle):
    """
    Gets the distance at a specific angle from lidar scan data.
    
    Args:
        scan_data: Array of 360 distance measurements
        body_angle: Angle in degrees (0-359)
        
    Returns:
        Tuple of (distance, angle) or None if no valid data
    """
    for angle in range(LIDAR_SCAN_POINTS):
        distance = scan_data[angle]

        if distance > 0 and body_angle == angle:
            return (distance, angle)

    return None

def get_cartesian(polar):
    """
    Converts polar coordinates to cartesian coordinates.
    
    Args:
        polar: Tuple of (distance, angle_in_degrees)
        
    Returns:
        Tuple of (x, y) cartesian coordinates
    """
    distance = polar[0]
    angle = polar[1]

    radians = angle * pi / 180.0
    x = distance * cos(radians)
    y = distance * sin(radians)
    return (x, y)

def get_distance(first, second):
    """
    Calculates Euclidean distance between two points in polar coordinates.
    
    Args:
        first: First point in polar coordinates (distance, angle)
        second: Second point in polar coordinates (distance, angle)
        
    Returns:
        Distance between the two points in millimeters
    """
    first = get_cartesian(first)
    second = get_cartesian(second)

    x1 = first[0]
    y1 = first[1]
    x2 = second[0]
    y2 = second[1]

    distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distance

def process_data(data):
    """
    Processes distance data by removing outliers using IQR method and calculating average.
    
    Args:
        data: List of distance measurements
        
    Returns:
        Tuple of (is_distanced, average_distance) where is_distanced is True if 
        average distance exceeds social distancing threshold
    """
    if not data:
        logging.warning("No data to process")
        return (False, 0)
    
    data = sorted(data)

    Q1 = np.percentile(data, 25, interpolation='midpoint')
    Q3 = np.percentile(data, 75, interpolation='midpoint')

    IQR = Q3 - Q1

    max_threshold = Q3 + (1.5 * IQR)
    min_threshold = Q1 - (1.5 * IQR)

    # Properly filter outliers by creating new list
    filtered_data = [value for value in data if min_threshold <= value <= max_threshold]
    
    # Check for division by zero
    if not filtered_data:
        logging.warning("All values were outliers, using original data")
        filtered_data = data
    
    # Use native Python arithmetic instead of FPGA
    total_sum = sum(int(num) for num in filtered_data)
    size = len(filtered_data)
    
    distance = total_sum / size
    distanced = distance > SOCIAL_DISTANCE_THRESHOLD_MM
        
    return (distanced, distance)

def print_output(distanced, occupancy):
    """
    Updates the output widgets with current occupancy and distance status.
    
    Args:
        distanced: Tuple of (is_distanced, distance_value)
        occupancy: Number of people detected
    """
    occupancy_out.clear_output() 
    with occupancy_out:
        display("Occupancy: " + str(occupancy))
    
    distance_out.clear_output() 
    with distance_out:
        display("Distance: " + str(distanced[1]))
        
    status_out.clear_output() 
    with status_out:
        if distanced[0]:
            display("Status: " + str("Distanced"))
        else:
            display("Status: " + str("Not Distanced"))
            
    sys.stdout.flush()

def collect_data(lidar, video_capture):
    """
    Continuously collects data from lidar and camera to monitor social distancing.
    
    Args:
        lidar: RPLidar instance
        video_capture: OpenCV VideoCapture instance
    """
    try:
        # Continuous processing loop for monitoring social distancing
        while True:
            distance_counter = 0
            occupancy = 0
            distance_data = []
            
            for scan in lidar.iter_scans(scan_type='express', max_buf_meas=False):
                _, frame = video_capture.read()

                if frame is not None:
                    frame = imutils.resize(frame, width=CAMERA_WIDTH)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    upper_body = haar_upper_body_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.05,
                        minNeighbors=4,
                        minSize=(50, 100),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    for (_, angle, distance) in scan:
                        scan_data[min(359, floor(angle))] = distance
                
                    occupancy = len(upper_body)

                    if len(upper_body) >= 2:
                        first_body_x = int(upper_body[0][0])
                        second_body_x = int(upper_body[1][0])

                        first_body_angle = map_x(first_body_x)
                        second_body_angle = map_x(second_body_x)

                        first_body_position = get_position(scan_data, first_body_angle)
                        second_body_position = get_position(scan_data, second_body_angle)

                        if first_body_position is not None and second_body_position is not None:
                            distance = get_distance(first_body_position, second_body_position)

                            distance_data.append(distance)
                            # Use native Python instead of FPGA
                            distance_counter += 1

                if distance_counter >= SCAN_SAMPLES_REQUIRED:
                    break

            distanced = process_data(distance_data)
            print_output(distanced, occupancy)
            # Continue loop instead of recursive call

    except KeyboardInterrupt:
        logging.info('Stopping.')
    except Exception as e:
        logging.error(f'Error in collect_data: {e}')
    finally:
        # Ensure cleanup happens
        if video_capture is not None:
            video_capture.release()

        if lidar is not None:
            try:
                lidar.stop_motor()
                lidar.stop()
                lidar.disconnect()
                logging.info('Stopped.')
            except Exception as e:
                logging.error(f'Error during cleanup: {e}')


def start_program():
    """
    Initializes and starts the distance validation program.
    Manages lidar and video capture resources with proper cleanup.
    """
    video_capture = None
    lidar = None
    
    try:
        video_capture = cv2.VideoCapture(0)
        lidar = RPLidar('/dev/ttyUSB0')
        
        # Pass both resources to collect_data
        collect_data(lidar, video_capture)

    except RPLidarException as e:
        logging.error(f'RPLidar exception: {e}')
    except Exception as e:
        logging.error(f'Error in start_program: {e}')
    finally:
        # Ensure cleanup happens
        if video_capture is not None:
            video_capture.release()

        if lidar is not None:
            try:
                lidar.stop_motor()
                lidar.stop()
                lidar.disconnect()
            except Exception as e:
                logging.error(f'Error during lidar cleanup: {e}')

start_program()