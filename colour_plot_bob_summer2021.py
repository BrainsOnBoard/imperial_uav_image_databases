#!/usr/bin/python3
import sys
sys.path.append('../navbench')

from time import time
import matplotlib.pyplot as plt
import numpy as np

import navbench as nb
from navbench import imgproc as ip

try:
    import pathos.multiprocessing as mp
except:
    mp = None
    print('WARNING: Could not find pathos.multiprocessing module')

@nb.caching.cache_result
def get_ridf_best_snaps(images, snapshots, step=1, parallel=None):
    if len(images) == 0 or len(snapshots) == 0:
        return np.array(())

    # Get a heading for a single image
    def get_best_snap_for_image(image):
        diffs = nb.ridf(image, snapshots, step=step)
        best_over_rot = np.min(diffs, axis=1)
        best_row = np.argmin(best_over_rot)
        return best_row

    def run_serial():
        return np.array([get_best_snap_for_image(image) for image in images])

    def run_parallel():
        with mp.Pool() as pool:
            return np.array(pool.map(get_best_snap_for_image, images))

    if parallel is None:
        # Module not installed
        if not mp:
            return run_serial()

        # Process in parallel if we have the module and there is a fair
        # amount of processing to be done
        num_ops = len(images) * len(snapshots) * images[0].size

        # This value was determined quasi-experimentally on my home machine -- AD
        if num_ops >= 120000:
            return run_parallel()
        return run_serial()

    if parallel:
        if mp:
            return run_parallel()

        print('WARNING: Parallel processing requested but pathos.multiprocessing module is not available')

    return run_serial()

PREPROC = ip.resize(360, 75)
TEST_DBS = ('2021-02-19_circle_slow_obstacle_1', )

db_train = nb.Database('datasets/unwrapped_2021-02-05_circle_slow_1')
train_entries = range(150, 500, 5)
db_test = nb.Database('datasets/unwrapped_' + TEST_DBS[0])
test_entries = range(670, 100, -5)

print('Maximum angle offset for training: %g°' % np.max(np.abs(db_train.heading)))

fig, ax = plt.subplots(len(TEST_DBS), figsize=(7, 7))
snapshots = db_train.read_images(entries=train_entries, preprocess=PREPROC)
images = db_test.read_images(entries=test_entries, preprocess=PREPROC)
snap_idxs = get_ridf_best_snaps(images, snapshots)

plt.plot(db_train.x[train_entries], db_train.y[train_entries], 'k--')
plt.scatter(db_test.x[test_entries], db_test.y[test_entries], c=snap_idxs / (len(snapshots) - 1),
            cmap='Spectral')
plt.plot(db_test.x[test_entries[0]], db_test.y[test_entries[0]], 'k+')

plt.axis('equal')
plt.colorbar()
plt.clim(0, 1)
plt.xlabel('$x$ (m)')
plt.ylabel('$y$ (m)')

plt.show()
