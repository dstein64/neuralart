from __future__ import print_function

import argparse
import copy
import os
import random
import sys
import warnings

from PIL import Image
import torch

version_txt = os.path.join(os.path.dirname(__file__), 'version.txt')
with open(version_txt, 'r') as f:
    __version__ = f.read().strip()

def get_devices():
    devices = ['cpu']
    # As of PyTorch 1.7.0, calling torch.cuda.is_available shows a warning ("...Found no NVIDIA
    # driver on your system..."). A related issue is reported in PyTorch Issue #47038.
    # Warnings are suppressed below to prevent a warning from showing when no GPU is available.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cuda_available = torch.cuda.is_available()
    if cuda_available and torch.cuda.device_count() > 0:
        devices.append('cuda')
        for idx in range(torch.cuda.device_count()):
            devices.append('cuda:{}'.format(idx))
    return tuple(devices)

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
           z=None,
           device='cpu'):
    if device not in get_devices():
        raise RuntimeError('Device {} not in available devices: {}'.format(
            device, ', '.join(get_devices())))

    cpu_rng_state = torch.get_rng_state()
    cuda_rng_states = []
    if torch.cuda.is_available():
        cuda_rng_states = [torch.cuda.get_rng_state(idx) for idx in range(torch.cuda.device_count())]

    if seed is None:
        seed = random.Random().randint(0, 2 ** 32 - 1)

    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    if ylim is None:
        ylim = copy.copy(xlim)

    if yres is None:
        yxscale = float(ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
        yres = int(yxscale * xres)

    x = torch.linspace(xlim[0], xlim[1], xres, device=device)
    y = torch.linspace(ylim[0], ylim[1], yres, device=device)
    grid = torch.meshgrid((y, x))

    inputs = torch.cat((grid[0].flatten().unsqueeze(1), grid[1].flatten().unsqueeze(1)), -1)

    if radius:
        inputs = torch.cat((inputs, torch.norm(inputs, 2, 1).unsqueeze(1)), -1)

    if z is not None:
        zrep = torch.tensor(z, dtype=inputs.dtype, device=device).repeat((inputs.shape[0], 1))
        inputs = torch.cat((inputs, zrep), -1)

    n_hidden_units = [units] * depth

    activations = inputs
    for units in n_hidden_units:
        if bias:
            bias_array = torch.ones((activations.shape[0], 1), device=device)
            activations = torch.cat((bias_array, activations), -1)
        hidden_layer_weights = torch.randn((activations.shape[1], units), device=device) * hidden_std
        activations = torch.tanh(torch.mm(activations, hidden_layer_weights))

    if bias:
        bias_array = torch.ones((activations.shape[0], 1), device=device)
        activations = torch.cat((bias_array, activations), -1)
    output_layer_weights = torch.randn((activations.shape[1], channels), device=device) * output_std
    output = torch.sigmoid(torch.mm(activations, output_layer_weights))
    output = output.reshape((yres, xres, channels))

    torch.set_rng_state(cpu_rng_state)
    for idx, cuda_rng_state in enumerate(cuda_rng_states):
        torch.cuda.set_rng_state(cuda_rng_state, idx)

    return (output.cpu() * 255).round().type(torch.uint8).numpy()

# ************************************************************
# * Command Line Interface
# ************************************************************

def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog='neuralart',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--version',
                        action='version',
                        version='neuralart {}'.format(__version__))
    parser.add_argument('--seed', type=int, help='RNG seed.')
    parser.add_argument('--xlim',
                        type=float,
                        nargs=2,
                        help='X limits.',
                        metavar=('MIN', 'MAX'),
                        default=[-1.0, 1.0])
    parser.add_argument('--ylim',
                        type=float,
                        nargs=2,
                        metavar=('MIN', 'MAX'),
                        help='Y limits. Defaults to xlim when not specified.')
    parser.add_argument('--xres', type=int, help='X resolution.', default=1024)
    parser.add_argument('--yres',
                        type=int,
                        help='Y resolution. When not specified, the value is calculated'
                             ' automatically based on xlim, ylim, and xres.')
    parser.add_argument('--units', type=int, help='Units per hidden layer.', default=16)
    parser.add_argument('--depth', type=int, help='Number of hidden layers.',default=8)
    parser.add_argument('--hidden-std',
                        type=float,
                        help='Standard deviation used to randomly initialize hidden layer weights.',
                        default=1.0)
    parser.add_argument('--output-std',
                        type=float,
                        help='Standard deviation used to randomly initialize output layer weights.',
                        default=1.0)
    parser.add_argument('--color-space',
                        choices=('rgb', 'bw'),
                        help='Select the color space (RGB or black-and-white).',
                        default='rgb')
    parser.add_argument('--no-radius',
                        action='store_false',
                        help='Disables radius input term.',
                        dest='radius')
    parser.add_argument('--no-bias',
                        action='store_false',
                        help='Disables bias terms.',
                        dest='bias')
    parser.add_argument('--device', default='cpu', choices=get_devices())
    parser.add_argument('--z', type=float, nargs='*')
    parser.add_argument('--no-verbose', action='store_false', dest='verbose')
    parser.add_argument('file', help='File path to save the PNG image.')
    args = parser.parse_args(argv[1:])
    return args


def main(argv=sys.argv):
    args = _parse_args(argv)
    if not args.file.lower().endswith('.png'):
        sys.stderr.write('Image file is missing PNG extension.\n')
    channels_lookup = {
        'rgb': 3,
        'bw': 1
    }
    seed = args.seed
    if seed is None:
        seed = random.randint(0, 2 ** 32 - 1)
    if args.verbose:
        print('Seed: {}'.format(seed))
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
        z=args.z,
        device=args.device
    )
    im = Image.fromarray(result.squeeze())
    im.save(args.file, 'png')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
