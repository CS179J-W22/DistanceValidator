import os
import cv2
import imutils
from math import cos, sin, pi, floor, sqrt
from rplidar import RPLidar, RPLidarException
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import ipywidgets as widgets 
import sys

lidar = None
haar_upper_body_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")
video_capture = cv2.VideoCapture(0)
print_counter = 0

occupancy_out = widgets.Output()
with occupancy_out:
    display("loading...")
display(occupancy_out)

status_out = widgets.Output()
with status_out:
    display("loading...")
display(status_out)

scan_data = [0]*360

def map_x(x_val):
    old_value = x_val

    old_min = 0
    old_max = 640

    new_min = 45
    new_max = 135

    new_value = None

    old_range = old_max - old_min

    if old_range == 0:
        new_value = new_min
    else:
        new_range = new_max - new_min
        old_diff = old_value - old_min
        new_value = ((old_diff * new_range) / old_range) + new_min

    return int(new_value)

def get_position(scan_data, body_angle):
    for angle in range(360):
        distance = scan_data[angle]

        if distance > 0 and body_angle == angle:
            return (distance, angle)

    return None

def get_cartesian(polar):
    distance = polar[0]
    angle = polar[1]

    radians = angle * pi / 180.0
    x = distance * cos(radians)
    y = distance * sin(radians)
    return (x, y)

def get_distance(first, second):
    first = get_cartesian(first)
    second = get_cartesian(second)

    x1 = first[0]
    y1 = first[1]
    x2 = second[0]
    y2 = second[1]

    distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distance

def process_data(data):
    data = sorted(data)

    Q1 = np.percentile(data, 25, interpolation = 'midpoint')
    Q3 = np.percentile(data, 75, interpolation = 'midpoint')

    IQR = Q3 - Q1

    max = Q3 + (1.5 * IQR)
    min = Q1 - (1.5 * IQR)

    for value in data:
        if value < min or value > max:
            value = None

    sum = 0
    size = 0
    for num in data:
        if num is not None:
            integer = int(num)
            sum += integer
            size += 1

    distance = sum / size
    distanced = False
    
    if distance > 1828.8:
        distanced = True
    else:
        distanced = False
        
    return (distanced, distance)

def print_output(distanced, occupancy):  
    status_out.clear_output() 
    with status_out:
        if distanced[0]:
            display("Status: " + str("Distanced"))
        else:
            display("Status: " + str("Not Distanced"))
            
    sys.stdout.flush()

def collect_data(lidar):
    try:
        
        distance_counter = 0
        occupancy = 0
        distance_data = []
        for scan in lidar.iter_scans(scan_type='express', max_buf_meas=False):
            _, frame = video_capture.read()

            if frame is not None:
                
                frame = imutils.resize(frame, width=640)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                upper_body = haar_upper_body_cascade.detectMultiScale(
                    gray,
                    scaleFactor = 1.05,
                    minNeighbors = 4,
                    minSize = (50, 100),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )

                for (_, angle, distance) in scan:
                    scan_data[min([359, floor(angle)])] = distance
            
                occupancy = len(upper_body)
                
                occupancy_out.clear_output() 
                with occupancy_out:
                    display("Occupancy: " + str(occupancy))

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
                        distance_counter += 1

            if distance_counter >= 5:
                break
                
        distanced = process_data(distance_data)
        print_output(distanced, occupancy)
        collect_data(lidar)


    except KeyboardInterrupt:
        print('Stopping.')

        video_capture.release()

        if lidar is not None:
            lidar.stop_motor()
            lidar.stop()
            lidar.disconnect()
            print('Stopped.')


def start_program():
    video_capture = cv2.VideoCapture(0)

    try:
        lidar = RPLidar('/dev/ttyUSB0')
        collect_data(lidar)

    except RPLidarException as e:
        video_capture.release()

        if lidar is not None:
            lidar.stop_motor()
            lidar.stop()
            lidar.disconnect()
            start_program()

start_program()