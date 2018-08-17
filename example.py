#!/usr/bin/env python

# Example usage of neuralart.py.

# Images can be converted to video with ffmpeg.
#   > ffmpeg -pattern_type glob -i "*.png" \
#            -c:v huffyuv -c:a libmp3lame -b:a 320k \
#            output.avi

from __future__ import print_function

import os
import sys

import numpy as np
import scipy

import neuralart

RENDER_SEED = 4
ITERATIONS = 200
RESOLUTION = 1024
Z_DIMS = 3

if len(sys.argv) != 2:
    sys.stderr.write("Usage: {} DIRECTORY\n".format(sys.argv[0]))
    sys.exit(1)

directory = sys.argv[1]
if not os.path.exists(directory):
    os.makedirs(directory)

zfill = len(str(ITERATIONS - 1))

z = np.array([-1.0] * Z_DIMS)
step_size = 2.0 / ITERATIONS
for x in range(ITERATIONS):
    result = neuralart.render(
        xres=RESOLUTION,
        seed=RENDER_SEED,
        channels=3,
        z=z
    )
    file = os.path.join(directory, str(x).zfill(zfill) + ".png")
    scipy.misc.imsave(file, result.squeeze(), format='png')
    z = z + step_size
