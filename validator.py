import os
import cv2
import imutils
from math import cos, sin, pi, floor, sqrt
from rplidar import RPLidar, RPLidarException
import matplotlib.pyplot as plt
import numpy as np

print('Starting.')

lidar = None
haar_upper_body_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")
video_capture = cv2.VideoCapture(0)

scan_data = [0]*360

def map_x(x_val):
    old_value = x_val

    old_min = 0
    old_max = 640

    new_min = 40
    new_max = 140

    new_value = None

    old_range = (old_max - old_min)

    if old_range == 0:
        new_value = new_min
    else:
        new_range = (new_max - new_min)
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return int(new_value)

def get_position(scan_data, body_angle):
    for angle in range(360):
        distance = scan_data[angle]

#         print(str(angle) + ", " + str(distance))

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

    Q1 = np.median(data[:5])
    Q3 = np.median(data[5:])

    interRange = Q3 - Q1

    max = Q3 + (1.5 * interRange)
    min = Q1 - (1.5 * interRange)

    for value in data:
        if value < min or value > max:
            value = None

    sum = 0
    size = 0
    for num in data:
        if num is not None:
            sum += int(num)
            size += 1

    print(sum/size)

def collect_data(lidar):
    try:
    #     print(lidar.info)
        distance_counter = 0
        distance_data = []
        for scan in lidar.iter_scans(scan_type='express', max_buf_meas=False):
            _, frame = video_capture.read()

            if frame is not None:
                frame = imutils.resize(frame, width=640)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                upper_body = haar_upper_body_cascade.detectMultiScale(
                    gray,
                    scaleFactor = 1.1,
                    minNeighbors = 2,
                    minSize = (50, 50),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )

#                 if len(upper_body) > 0:
#                     print(upper_body)

                for (_, angle, distance) in scan:
                    scan_data[min([359, floor(angle)])] = distance

                if len(upper_body) >= 2:
                    first_body_x = int(upper_body[0][0])
                    second_body_x = int(upper_body[1][0])

                    first_body_angle = map_x(first_body_x)
                    second_body_angle = map_x(second_body_x)

#                         print(str(first_body_angle) + ", " + str(second_body_angle))

                    first_body_position = get_position(scan_data, first_body_angle)
                    second_body_position = get_position(scan_data, second_body_angle)

#                         if first_body_position is not None and second_body_position is not None:
#                             print(str(first_body_position) + ", " + str(second_body_position))

                    if first_body_position is not None and second_body_position is not None:
                        distance = get_distance(first_body_position, second_body_position)

                        distance_data.append(distance)
                        distance_counter += 1

#                         print(distance_data)

            if distance_counter >= 10:
                break

        process_data(distance_data)

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
    try:
        lidar = RPLidar('/dev/ttyUSB0')
        collect_data(lidar)

    except RPLidarException as e:
        if lidar is not None:
            lidar.stop_motor()
            lidar.stop()
            lidar.disconnect()
            start_program()

start_program()
