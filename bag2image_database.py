#!/usr/bin/python3

'''

This script extracts images from ros bag files and stores them in specified folder
with image names as corresponding timestamps.

This script is the modified version of bag_to_csv script written by Nick Speal in May 2013 at McGill University's Aerospace Mechatronics Laboratory
www.speal.ca

'''

import rosbag
import sys
import csv
import time
import string
import os  # for file management make directory
import shutil  # for file management, copy file
from cv_bridge import CvBridge
import cv2
import rospy
import numpy as np
import math


assert len(sys.argv) == 2

bridge = CvBridge()

bag = rosbag.Bag(sys.argv[1])
bagName = bag.filename

print('Processing ' + bagName)

# create a new directory
folder = bagName.rsplit('.', 1)[0]

try:  # else already exists
    os.makedirs(folder)
except:
    pass

times = []
poses = np.empty((0, 6))
for _, msg, t in bag.read_messages('/vicon/dd/odom'):
    times.append(t.to_sec())
    pose = msg.pose.pose
    pose_arr = [pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z]
    poses = np.concatenate((poses, [pose_arr]))

topics = set()
start_time = bag.get_start_time()
with open(os.path.join(folder, 'database_entries.csv'), 'w') as csv:
    csv.write('Timestamp [ms], X [mm], Y [mm], Z [mm], Heading [degrees], ' +
              'Pitch [degrees], Roll [degrees], Filename\n')

    count = 0
    for _, msg, t in bag.read_messages('/kodak/image_raw'):
        time = t.to_sec()
        try:
            gt_idx = next(i for i, val in enumerate(times) if val > time)
        except StopIteration:
            break

        csv.write('%f, ' % (1000 * (time - start_time)))

        lt_idx = gt_idx - 1
        rng = times[gt_idx] - times[lt_idx]
        prop = (time - times[lt_idx]) / rng
        p = poses[lt_idx, :] + prop * (poses[gt_idx, :] - poses[lt_idx, :])
        csv.write('%f, %f, %f, ' % (p[0] * 1000, p[1] * 1000, p[2] * 1000))
        csv.write('%f, %f, %f, ' % (math.degrees(p[3]), math.degrees(p[4]), math.degrees(p[5])))

        filename = 'image%05d.jpg' % count
        csv.write(filename + '\n')
        count += 1
        im = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        assert cv2.imwrite(os.path.join(folder, filename), im)
