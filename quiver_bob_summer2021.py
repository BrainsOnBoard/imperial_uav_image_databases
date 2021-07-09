#!/usr/bin/python3
import sys
sys.path.append('../navbench')

from time import time
import matplotlib.pyplot as plt
import numpy as np

import navbench as nb
from navbench import imgproc as ip

PREPROC = ip.resize(360, 75)
TEST_DBS = ('2021-02-05_circle_slow_2', ) #'2021-02-05_circle_slow_2', '2021-02-19_circle_fast_obstacle_2', '2021-02-19_circle_slow_obstacle_1', '2021-02-19_circle_slow_obstacle_2')

db_train = nb.Database('datasets/unwrapped_2021-02-05_circle_slow_1')
dbs_test = (nb.Database('datasets/unwrapped_' + name) for name in TEST_DBS)

print('Maximum angle offset for training: %g°' % np.max(np.abs(db_train.heading)))

_, axes = plt.subplots(len(TEST_DBS), figsize=(7, 7))
for ax, db_test in zip([axes], dbs_test):
    test_entries = range(0, len(db_test), 5)
    snapshots = db_train.read_images(entries=range(0, len(db_train), 5), preprocess=PREPROC)
    images = db_test.read_images(entries=test_entries, preprocess=PREPROC)

    headings = nb.get_ridf_headings(images, snapshots)
    u = np.cos(headings)
    v = np.sin(headings)
    ax.quiver(db_test.x[test_entries], db_test.y[test_entries], u, v)
    ax.axis('equal')
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')

plt.savefig('quiver_bob_summer2021.svg')
plt.show()
