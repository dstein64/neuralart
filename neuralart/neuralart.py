from __future__ import print_function

import argparse
import copy
import os
import sys

import numpy as np
import scipy.misc

version_txt = os.path.join(os.path.dirname(__file__), 'version.txt')
with open(version_txt, 'r') as f:
    __version__ = f.read().strip()

# ************************************************************
# * Core
# ************************************************************

def render(seed=None,
           xlim=[-1.0, 1.0],
           ylim=None,
           xres=1024,
           yres=None,
           units=16,
           depth=8,
           hidden_std=1.0,
           output_std=1.0,
           channels=3,
           radius=True,
           bias=True,
           z=None):
    if seed is None:
        seed = np.random.RandomState().randint(2 ** 32, dtype=np.uint32)

    rng = np.random.RandomState(seed=seed)

    if not ylim:
        ylim = copy.copy(xlim)

    if not yres:
        yxscale = float(ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
        yres = int(yxscale * xres)

    x = np.linspace(xlim[0], xlim[1], xres)
    y = np.linspace(ylim[0], ylim[1], yres)
    grid = np.meshgrid(x, y)

    inputs = np.vstack((grid[0].flatten(), grid[1].flatten())).T

    if radius:
        inputs = np.hstack((inputs, np.linalg.norm(inputs, axis=1)[:, np.newaxis]))

    if z is not None:
        inputs = np.hstack((inputs, np.matlib.repmat(z, inputs.shape[0], 1)))

    n_hidden_units = [units] * depth

    activations = inputs
    for units in n_hidden_units:
        if bias:
            activations = np.hstack(
                (np.ones((activations.shape[0], 1)), activations))
        hidden_layer_weights = rng.normal(
            scale=hidden_std, size=(activations.shape[1], units))
        activations = np.tanh(np.dot(activations, hidden_layer_weights))

    if bias:
        activations = np.hstack(
            (np.ones((activations.shape[0], 1)), activations))
    output_layer_weights = rng.normal(
        scale=output_std, size=(activations.shape[1], channels))

    logits = np.dot(activations, output_layer_weights)
    output = 1.0 / (1.0 + np.exp(-logits))

    output = output.reshape((yres, xres, channels))

    return output

# ************************************************************
# * Command Line Interface
# ************************************************************

def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="neuralart",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--version",
                        action="version",
                        version="neuralart {}".format(__version__))
    parser.add_argument("--seed", type=int, help="RNG seed.")
    parser.add_argument("--xlim",
                        type=float,
                        nargs=2,
                        help="X limits.",
                        metavar=('MIN', 'MAX'),
                        default=[-1.0, 1.0])
    parser.add_argument("--ylim",
                        type=float,
                        nargs=2,
                        metavar=('MIN', 'MAX'),
                        help="Y limits. Defaults to xlim when not specified.")
    parser.add_argument("--xres", type=int, help="X resolution.", default=1024)
    parser.add_argument("--yres",
                        type=int,
                        help="Y resolution. When not specified, the value is calculated"
                             " automatically based on xlim, ylim, and xres.")
    parser.add_argument("--units", type=int, help="Units per hidden layer.", default=16)
    parser.add_argument("--depth", type=int, help="Number of hidden layers.",default=8)
    parser.add_argument("--hidden-std",
                        type=float,
                        help="Standard deviation used to randomly initialize hidden layer weights.",
                        default=1.0)
    parser.add_argument("--output-std",
                        type=float,
                        help="Standard deviation used to randomly initialize output layer weights.",
                        default=1.0)
    parser.add_argument("--color-space",
                        choices=('rgb', 'bw'),
                        help="Select the color space (RGB or black-and-white).",
                        default='rgb')
    parser.add_argument("--no-radius",
                        action="store_false",
                        help="Disables radius input term.",
                        dest="radius")
    parser.add_argument("--no-bias",
                        action="store_false",
                        help="Disables bias terms.",
                        dest="bias")
    parser.add_argument("--z", type=float, nargs="*")
    parser.add_argument("--no-verbose", action='store_false', dest='verbose')
    parser.add_argument("file", help="File path to save the PNG image.")
    args = parser.parse_args(argv[1:])
    return args


def main(argv=sys.argv):
    args = _parse_args(argv)
    if not args.file.lower().endswith(".png"):
        sys.stderr.write("Image file is missing PNG extension.\n")
    channels_lookup = {
        'rgb': 3,
        'bw': 1
    }
    seed = args.seed
    if seed is None:
        seed = np.random.RandomState().randint(2 ** 32, dtype=np.uint32)
    if args.verbose:
        print("Seed: {}".format(seed))
    result = render(
        seed=seed,
        xlim=args.xlim,
        ylim=args.ylim,
        xres=args.xres,
        yres=args.yres,
        units=args.units,
        depth=args.depth,
        hidden_std=args.hidden_std,
        output_std=args.output_std,
        channels=channels_lookup[args.color_space],
        radius=args.radius,
        bias=args.bias,
        z=args.z
    )
    scipy.misc.imsave(args.file, result.squeeze(), format='png')
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
