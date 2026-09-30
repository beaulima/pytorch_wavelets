""" Time a forward and inverse transform against the numpy implementations.

Compares pytorch_wavelets with PyWavelets (DWT) or with the ``dtcwt`` package
(DTCWT), on a batch of single-channel images. For example::

    python benchmarks/compare_numpy.py torch dwt --device cpu
    python benchmarks/compare_numpy.py numpy dwt
    python benchmarks/compare_numpy.py torch dtcwt -j 3

The numpy DTCWT needs the ``dtcwt`` package, which the ``test`` extra
installs.
"""
import argparse
import timeit

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('method', choices=['torch', 'numpy'],
                    help='Method to use to calculate the transform')
parser.add_argument('xfm', choices=['dwt', 'dtcwt'],
                    help='which transform to use')
parser.add_argument('-j', type=int, default=2,
                    help='number of scales of transform to do')
parser.add_argument('-s', '--size', default=128, type=int,
                    help='spatial size of input')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                    help='which device to test (torch only)')
parser.add_argument('--wave', default='db4',
                    help='which wavelet to use (DWT only)')
parser.add_argument('--batch', default=16, type=int,
                    help='Number of images in parallel')
parser.add_argument('-n', '--number', default=5, type=int,
                    help='Number of runs to average over')

TORCH_SETUP = {
    'dwt': """
from pytorch_wavelets import DWT, IDWT
xfm = DWT(J={J}, wave='{wave}').to('{dev}')
ifm = IDWT(wave='{wave}').to('{dev}')""",
    'dtcwt': """
from pytorch_wavelets import DTCWTForward, DTCWTInverse
xfm = DTCWTForward(J={J}).to('{dev}')
ifm = DTCWTInverse().to('{dev}')""",
}

NUMPY = {
    'dwt': ('xfm(x)', """
import numpy as np
import pywt
x = np.random.randn(*{sz})
xfm = lambda a: pywt.waverec2(
    pywt.wavedec2(a, '{wave}', level={J}, mode='reflect'), '{wave}',
    mode='reflect')"""),
    # The dtcwt package transforms one 2D image at a time.
    'dtcwt': ("""
for b in x:
    for c in b:
        xfm.inverse(xfm.forward(c, nlevels={J}))""", """
import numpy as np
import dtcwt
x = np.random.randn(*{sz})
xfm = dtcwt.Transform2d(biort='near_sym_a', qshift='qshift_a')"""),
}


def main():
    args = parser.parse_args()
    size = (args.batch, 1, args.size, args.size)
    fmt = dict(sz=size, J=args.j, wave=args.wave, dev=args.device)

    if args.method == 'torch':
        # CUDA kernels run asynchronously: without the synchronize, the timer
        # would stop as soon as the work was queued.
        sync = ('torch.cuda.synchronize()' if args.device == 'cuda'
                else 'pass')
        stmt = 'y = ifm(xfm(x)); ' + sync
        setup = ('import torch\nx = torch.randn(*{sz}).to(\'{dev}\')'
                 + TORCH_SETUP[args.xfm]).format(**fmt)
        # One untimed pass, so the timing leaves out CUDA initialisation.
        setup += '\n' + stmt
    else:
        stmt, setup = (s.format(**fmt) for s in NUMPY[args.xfm])

    t = timeit.Timer(stmt, setup=setup).timeit(number=args.number)
    print('{} {} on {}: {} run average is {:.3f}s'.format(
        args.method, args.xfm, size, args.number, t / args.number))


if __name__ == "__main__":
    main()
