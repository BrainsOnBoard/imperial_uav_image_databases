# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
sys.path.append('../navbench')

from time import time

import navbench as nb
from navbench import improc as ip
import numpy as np

import matplotlib.pyplot as plt

PREPROC = ip.resize(80, 10)

db_train = nb.Database('datasets/unwrapped_2021-02-05_circle_slow_1')
db_test = nb.Database('datasets/unwrapped_2021-02-05_circle_slow_2')

snapshots = db_train.read_images(entries=range(0, len(db_train), 5), preprocess=PREPROC)
images = db_test.read_images(entries=range(0, len(db_test), 10), preprocess=PREPROC)

def run_test(num_ims, parallel):
    print('num_ims=%d' % num_ims)
    t0 = time()
    nb.get_ridf_headings_no_cache(images[:num_ims], snapshots[:10], parallel=parallel)
    return time() - t0

num_ims = range(0, 70, 4)
print('serial')
res_serial = [run_test(n, False) for n in num_ims]
print('parallel')
res_parallel = [run_test(n, True) for n in num_ims]
print('heuristic')
res_heur = [run_test(n, None) for n in num_ims]

x = np.array(num_ims) * images[0].size * 10
plt.plot(x, res_serial, x, res_parallel, x, res_heur)
plt.show()
