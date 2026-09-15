import torch.nn as nn
import pywt
import pytorch_wavelets.dwt.lowlevel as lowlevel
import torch


class DWTForward(nn.Module):
    """ Performs a 2d DWT Forward decomposition of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        separable (bool): If true (the default), filter the rows and columns
            with two 1d convolutions. If false, use a single non-separable 2d
            convolution with the outer-product kernels. Both give the same
            coefficients; which is faster depends on the wavelet, the input
            size and the hardware.
        """
    def __init__(self, J=1, wave='db1', mode='zero', separable=True):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]
            else:
                raise ValueError(
                    "Expected a 2- or 4-tuple of filters, got a tuple "
                    "of length {}".format(len(wave)))

        # Prepare the filters
        self.separable = separable
        if separable:
            filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
            self.register_buffer('h0_col', filts[0])
            self.register_buffer('h1_col', filts[1])
            self.register_buffer('h0_row', filts[2])
            self.register_buffer('h1_row', filts[3])
        else:
            filts = lowlevel.prep_filt_afb2d_nonsep(
                h0_col, h1_col, h0_row, h1_row)
            self.register_buffer('h', filts)
        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the DWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            (yl, yh)
                tuple of lowpass (yl) and bandpass (yh) coefficients.
                yh is a list of length J with the first entry
                being the finest scale coefficients. yl has shape
                :math:`(N, C_{in}, H_{in}', W_{in}')` and yh has shape
                :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. The new
                dimension in yh iterates over the LH, HL and HH coefficients.

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.
        """
        assert x.ndim == 4, "Can only handle 4d inputs (N, C, H, W)"
        yh = []
        ll = x
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel transform
        for j in range(self.J):
            # Do 1 level of the transform
            if self.separable:
                ll, high = lowlevel.AFB2D.apply(
                    ll, self.h0_col, self.h1_col, self.h0_row, self.h1_row,
                    mode)
            else:
                ll, high = lowlevel.AFB2D_nonsep.apply(ll, self.h, mode)
            yh.append(high)

        return ll, yh


class DWTInverse(nn.Module):
    """ Performs a 2d DWT Inverse reconstruction of an image

    Args:
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (h0, h1) or (h0_col, h1_col, h0_row, h1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme
        separable (bool): If true (the default), filter the rows and columns
            with two 1d convolutions. If false, use a single non-separable 2d
            convolution with the outer-product kernels. Must match the
            `separable` setting used for the forward transform's padding to
            line up.
    """
    def __init__(self, wave='db1', mode='zero', separable=True):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]
            else:
                raise ValueError(
                    "Expected a 2- or 4-tuple of filters, got a tuple "
                    "of length {}".format(len(wave)))
        # Prepare the filters
        self.separable = separable
        if separable:
            filts = lowlevel.prep_filt_sfb2d(g0_col, g1_col, g0_row, g1_row)
            self.register_buffer('g0_col', filts[0])
            self.register_buffer('g1_col', filts[1])
            self.register_buffer('g0_row', filts[2])
            self.register_buffer('g1_row', filts[3])
        else:
            filts = lowlevel.prep_filt_sfb2d_nonsep(
                g0_col, g1_col, g0_row, g1_row)
            self.register_buffer('g', filts)
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (yl, yh): tuple of lowpass and bandpass coefficients, where:
              yl is a lowpass tensor of shape :math:`(N, C_{in}, H_{in}',
              W_{in}')` and yh is a list of bandpass tensors of shape
              :math:`list(N, C_{in}, 3, H_{in}'', W_{in}'')`. I.e. should match
              the format returned by DWTForward

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Note:
            :math:`H_{in}', W_{in}', H_{in}'', W_{in}''` denote the correctly
            downsampled shapes of the DWT pyramid.

        Note:
            Can have None for any of the highpass scales and will treat the
            values as zeros (not in an efficient way though).
        """
        yl, yh = coeffs
        assert yl.ndim == 4, "Can only handle 4d inputs (N, C, H, W)"
        ll = yl
        mode = lowlevel.mode_to_int(self.mode)

        # Do a multilevel inverse transform
        for h in yh[::-1]:
            if h is None:
                h = torch.zeros(ll.shape[0], ll.shape[1], 3, ll.shape[-2],
                                ll.shape[-1], device=ll.device)

            # 'Unpad' added dimensions
            if ll.shape[-2] > h.shape[-2]:
                ll = ll[...,:-1,:]
            if ll.shape[-1] > h.shape[-1]:
                ll = ll[...,:-1]
            if self.separable:
                ll = lowlevel.SFB2D.apply(
                    ll, h, self.g0_col, self.g1_col, self.g0_row, self.g1_row,
                    mode)
            else:
                ll = lowlevel.SFB2D_nonsep.apply(ll, h, self.g, mode)
        return ll


class SWTForward(nn.Module):
    """ Performs a 2d Stationary wavelet transform (or undecimated wavelet
    transform) of an image

    Args:
        J (int): Number of levels of decomposition
        wave (str or pywt.Wavelet): Which wavelet to use. Can be a string to
            pass to pywt.Wavelet constructor, can also be a pywt.Wavelet class,
            or can be a two tuple of array-like objects for the analysis low and
            high pass filters.
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme. PyWavelets uses only periodization so we use this
            as our default scheme.
        """
    def __init__(self, J=1, wave='db1', mode='periodization'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            h0_col, h1_col = wave.dec_lo, wave.dec_hi
            h0_row, h1_row = h0_col, h1_col
        else:
            if len(wave) == 2:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = h0_col, h1_col
            elif len(wave) == 4:
                h0_col, h1_col = wave[0], wave[1]
                h0_row, h1_row = wave[2], wave[3]
            else:
                raise ValueError(
                    "Expected a 2- or 4-tuple of filters, got a tuple "
                    "of length {}".format(len(wave)))

        # Prepare the filters
        filts = lowlevel.prep_filt_afb2d(h0_col, h1_col, h0_row, h1_row)
        self.register_buffer('h0_col', filts[0])
        self.register_buffer('h1_col', filts[1])
        self.register_buffer('h0_row', filts[2])
        self.register_buffer('h1_row', filts[3])

        self.J = J
        self.mode = mode

    def forward(self, x):
        """ Forward pass of the SWT.

        Args:
            x (tensor): Input of shape :math:`(N, C_{in}, H_{in}, W_{in})`

        Returns:
            List of coefficients for each scale. Each coefficient has
            shape :math:`(N, C_{in}, 4, H_{in}, W_{in})` where the extra
            dimension stores the 4 subbands for each scale. The ordering in
            these 4 coefficients is: (A, H, V, D) or (ll, lh, hl, hh).
        """
        ll = x
        coeffs = []
        # Do a multilevel transform
        filts = (self.h0_col, self.h1_col, self.h0_row, self.h1_row)
        for j in range(self.J):
            # Do 1 level of the transform
            y = lowlevel.afb2d_atrous(ll, filts, self.mode, 2**j)
            # afb2d_atrous stacks the 4 subbands along the channel dimension,
            # giving (N, C*4, H, W). Split that back out into its own axis so
            # that y[:,:,0] selects the lowpass rather than a row of pixels.
            s = y.shape
            y = y.reshape(s[0], -1, 4, s[-2], s[-1])
            coeffs.append(y)
            ll = y[:,:,0]

        return coeffs


class SWTInverse(nn.Module):
    """ Performs a 2d inverse Stationary Wavelet Transform (or inverse
    undecimated wavelet transform) - the inverse of :class:`SWTForward`.

    Args:
        wave (str or pywt.Wavelet or tuple(ndarray)): Which wavelet to use.
            Can be:
            1) a string to pass to pywt.Wavelet constructor
            2) a pywt.Wavelet class
            3) a tuple of numpy arrays, either (g0, g1) or (g0_col, g1_col, g0_row, g1_row)
        mode (str): 'zero', 'symmetric', 'reflect' or 'periodization'. The
            padding scheme. Must match the one used for the forward transform.

    Note:
        Perfect reconstruction only holds for periodic extension
        ('periodization' or 'periodic', the default), because the undecimated
        reconstruction identity :math:`[G_0(z)H_0(z) + G_1(z)H_1(z)]/2 = 1`
        assumes circular convolution. With the other padding schemes the
        interior is still recovered exactly, but samples within roughly a
        filter length of the border are not. PyWavelets has the same
        restriction - its ``swt2`` is periodic only.
    """
    def __init__(self, wave='db1', mode='periodization'):
        super().__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            g0_col, g1_col = wave.rec_lo, wave.rec_hi
            g0_row, g1_row = g0_col, g1_col
        else:
            if len(wave) == 2:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = g0_col, g1_col
            elif len(wave) == 4:
                g0_col, g1_col = wave[0], wave[1]
                g0_row, g1_row = wave[2], wave[3]
            else:
                raise ValueError(
                    "Expected a 2- or 4-tuple of filters, got a tuple "
                    "of length {}".format(len(wave)))

        # prep_filt_afb1d time reverses the filters, which is what
        # sfb1d_atrous wants so that conv2d performs a convolution.
        filts = lowlevel.prep_filt_afb2d(g0_col, g1_col, g0_row, g1_row)
        self.register_buffer('g0_col', filts[0])
        self.register_buffer('g1_col', filts[1])
        self.register_buffer('g0_row', filts[2])
        self.register_buffer('g1_row', filts[3])
        self.mode = mode

    def forward(self, coeffs):
        """
        Args:
            coeffs (list): list of coefficient tensors of shape
                :math:`(N, C_{in}, 4, H_{in}, W_{in})`, finest scale first, as
                returned by :class:`SWTForward`. Only the lowpass of the
                coarsest scale is used; the finer lowpasses are redundant and
                are recomputed.

        Returns:
            Reconstructed input of shape :math:`(N, C_{in}, H_{in}, W_{in})`
        """
        filts = (self.g0_col, self.g1_col, self.g0_row, self.g1_row)
        ll = coeffs[-1][:, :, 0]

        # Coarsest scale first, mirroring the forward transform's dilations.
        for j in range(len(coeffs) - 1, -1, -1):
            step = coeffs[j]
            ll = lowlevel.sfb2d_atrous(
                ll, step[:, :, 1], step[:, :, 2], step[:, :, 3], filts,
                mode=self.mode, dilation=2**j)

        return ll
