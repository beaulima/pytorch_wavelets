""" Workloads for profiling the DTCWT, forward, inverse or both.

Each run repeats the transform five times on a (batch, 5, size, size) input,
backward pass included unless --no_grad is given, so that the result can be
fed to a profiler. For example::

    python benchmarks/profile_dtcwt.py -f -j 2
    nsys profile -o dtcwt_j2 python benchmarks/profile_dtcwt.py -j 2
    python -m torch.utils.bottleneck benchmarks/profile_dtcwt.py -j 2

The --conv, --ref and --dwt options run the reference workloads that
docs/speed.rst compares against: an 11x11 convolution, a DTCWT-sized FFT
convolution, and a separable DWT.
"""
import argparse

import torch
import torch.nn.functional as F

from pytorch_wavelets import DTCWTForward, DTCWTInverse, DWTForward

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--no_grad', action='store_true',
                    help='Dont calculate the gradients')
parser.add_argument('--ref', action='store_true',
                    help='Compare to doing the DTCWT with ffts')
parser.add_argument('-c', '--convolution', action='store_true',
                    help='Profile an 11x11 convolution')
parser.add_argument('--dwt', action='store_true',
                    help='Profile dwt instead of dtcwt')
parser.add_argument('-f', '--forward', action='store_true',
                    help='Only do forward transform (default is fwd and inv)')
parser.add_argument('-i', '--inverse', action='store_true',
                    help='Only do inverse transform (default is fwd and inv)')
parser.add_argument('-j', type=int, default=2,
                    help='number of scales of transform to do')
parser.add_argument('--no_hp', action='store_true',
                    help='Skip the first scale highpasses (skip_hps)')
parser.add_argument('-s', '--size', default=128, type=int,
                    help='spatial size of input')
parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                    help='which device to test')
parser.add_argument('--batch', default=16, type=int,
                    help='Number of images in parallel')


def _input(size, no_grad, dev):
    return torch.randn(*size, device=dev, requires_grad=not no_grad)


def forward(size, no_grad, J, no_hp=False, dev='cuda'):
    x = _input(size, no_grad, dev)
    xfm = DTCWTForward(J=J, skip_hps=no_hp, o_dim=1).to(dev)
    for _ in range(5):
        Yl, Yh = xfm(x)
        if not no_grad:
            Yl.backward(torch.ones_like(Yl))


def inverse(size, no_grad, J, no_hp=False, dev='cuda'):
    yl = torch.randn(size[0], size[1], size[2] >> (J-1), size[3] >> (J-1),
                     device=dev, requires_grad=not no_grad)
    yh = [torch.randn(size[0], size[1], 6, size[2] >> j, size[3] >> j, 2,
                      device=dev, requires_grad=not no_grad)
          for j in range(1, J+1)]
    ifm = DTCWTInverse().to(dev)
    for _ in range(5):
        Y = ifm((yl, yh))
        if not no_grad:
            Y.backward(torch.ones_like(Y))


def end_to_end(size, no_grad, J, no_hp=False, dev='cuda'):
    x = _input(size, no_grad, dev)
    xfm = DTCWTForward(J=J, skip_hps=no_hp).to(dev)
    ifm = DTCWTInverse().to(dev)
    for _ in range(5):
        # The forward runs inside the loop: each backward consumes its graph.
        Y = ifm(xfm(x))
        if not no_grad:
            Y.backward(torch.ones_like(Y))


def reference_conv(size, no_grad=False, dev='cuda'):
    x = _input(size, no_grad, dev)
    w = torch.randn(10, size[1], 11, 11, device=dev)
    y = F.conv2d(x, w, padding=5)
    if not no_grad:
        y.backward(torch.ones_like(y))


def reference_fftconv(size, J, no_grad=False, dev='cuda'):
    """ Filter with 12J+1 random frequency responses by FFT, roughly the work
    of a DTCWT with wavelets 9J samples across. """
    x = _input(size, no_grad, dev)
    sz = 9*J
    xp = F.pad(x, (0, sz-1, 0, sz-1))
    FX = torch.fft.rfft2(xp).unsqueeze(2)
    FW = torch.randn(1, 1, 12*J+1, *FX.shape[-2:], dtype=FX.dtype, device=dev)
    FY = (FX * FW).flatten(1, 2)
    Y = torch.fft.irfft2(FY, s=xp.shape[-2:])
    if not no_grad:
        Y.backward(torch.ones_like(Y))


def separable_dwt(size, J, no_grad=False, dev='cuda'):
    x = _input(size, no_grad, dev)
    xfm = DWTForward(J, wave='db5', mode='zero').to(dev)
    for _ in range(5):
        yl, yh = xfm(x)
        if not no_grad:
            yh[0].backward(torch.ones_like(yh[0]))


def main():
    args = parser.parse_args()
    size = (args.batch, 5, args.size, args.size)

    if args.ref:
        print('Running dtcwt with FFTs')
        reference_fftconv(size, args.j, args.no_grad, args.device)
    elif args.convolution:
        print('Running 11x11 convolution')
        reference_conv(size, args.no_grad, args.device)
    elif args.dwt:
        print('Running separable dwt')
        separable_dwt(size, args.j, args.no_grad, args.device)
    elif args.forward:
        print('Running forward transform')
        forward(size, args.no_grad, args.j, args.no_hp, args.device)
    elif args.inverse:
        print('Running inverse transform')
        inverse(size, args.no_grad, args.j, args.no_hp, args.device)
    else:
        print('Running end to end')
        end_to_end(size, args.no_grad, args.j, args.no_hp, args.device)

    if args.device == 'cuda':
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
