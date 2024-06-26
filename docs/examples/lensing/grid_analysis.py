# Code copied and concatenated from the Miko project https://arxiv.org/abs/2312.08934 and https://arxiv.org/abs/2304.01387

from functools import partial

import jax.numpy as jnp
import jax.numpy.fft as fft
import matplotlib.pyplot as plt
import numpy as np
from jax import jit, vmap

import jax.numpy as jnp
import numpy as np

import matplotlib.pyplot as plt
import numpy as np


class Grid:
    def __init__(
        self,
        xlim,
        ylim,
        xn: int,
        yn: int,
        unit: str = "deg",
        v: bool = False,
    ):
        """
        Initialize grid parameters.

        Parameters
        ----------
        xlim, ylim : list or np.ndarray
            [lower_bound, upper_bound] in real space.
        xn, yn : int
            Number of grid points along each axis.
        unit : str, optional
            Unit of the bounds, 'deg' or 'rad'. Defaults to 'deg'.
        v : bool, optional
            Verbosity flag. Defaults to False.

        Attributes
        ----------
        Real space:
            xlim, ylim : ndarray
                Array of length 2 in radians.
            xspan, yspan : float
                Spans of the x-axis and y-axis in radians.
            xn, yn : int
                Number of grid points used in each axis. These should be even integers.
            xres, yres : float
                Resolutions of the x-axis and y-axis in radians.
            fsky : float
                Fraction of the sky covered by the grid (unitless).
            xcoord_boundry : ndarray
                1D numpy array.
            ycoord_boundry : ndarray
                1D numpy array.
            xcoord, ycoord : ndarray
                1D numpy arrays.
            x, y : ndarray
                2D array that records the coordinate value of real space grid point centers.
                Shape: rmap_shape.
            rmap_shape : tuple
                Shape of the real space grid.

        Ell space:
            lxlim, lylim : ndarray
                Array of length 2 in angular frequency (\ell).
            Nyquist : float
                The angular Nyquist frequency. This is the same in x and y direction since we will force the x/y resolution to be the same.
            lxn, lyn : int
                Number of grid points used in each axis.
            lxres, lyres : float
                Resolutions of the x-axis and y-axis in ell space.
            lxcoord, lycoord : ndarray
                Sorted 1D numpy arrays for the numpy rfft coordinates.
            lx, ly : ndarray
                2D array that records the coordinate value of Fourier space grid point centers.
                Shape: lmap_shape.
            lnrm : ndarray
                2D array that records the |\ell| values of the Fourier space grid point centers.
                Shape: lmap_shape.
            lmap_shape : tuple
                Shape of the ell space grid.

            XPS_conversion_ell2real2d: float
                If we want to realize the cosmological XPS = C(|\ell|) on the 2d grid by drawing random realizations on the ell grid,
                then the variance of the random variable at distance |\ell| from the origin should be C(|\ell|) / XPS_conversion_ell2real.
                Equivalently, if we measure XPS on the ell grid to be f(\ell), then we need to compare f(\ell) * XPS_conversion_ell2real
                with the cosmological C(|\ell|).

        """
        # check xlim and ylim must be list or np.array
        if not isinstance(xlim, (list, np.ndarray)) or not isinstance(ylim, (list, np.ndarray)):
            raise TypeError("'xlim' and 'ylim' must be list or np.ndarray")

        if not isinstance(xn, int) or not isinstance(yn, int):
            raise TypeError("'xn' and 'yn' must be integers")

        if xn % 2 != 0 or yn % 2 != 0:
            raise ValueError("'xn' and 'yn' must be even integers")

        if unit not in ["deg", "rad"]:
            raise ValueError("'unit' must be either 'deg' or 'rad'")

        # for every attribute stored, the unit of any angular quantity is in radians.
        if unit == "deg":
            xlim = np.radians(xlim)
            ylim = np.radians(ylim)

        self._set_real_space_parameters(xlim, ylim, xn, yn, v)
        self._set_ell_space_parameters(v)
        # self.kernel_KS1, self.kernel_KS2 = self.kernel_lspKS(self.lx, self.ly, check=True)
        # self.kernel_pixel = self.kernel_lsppixeltophat(self.xres, self.yres, self.lx, self.ly)
        if v:
            self.print_summary()

        # plotting
        self.plotter = PlotGrid(self.x, self.y, self.lx, self.ly)
        return

    def __repr__(self):
        return "grid(xlim=%r, xnpix=%r, ylim=%r, ynpix=%r) units in [rad]" % (
            self.xlim,
            self.xn,
            self.ylim,
            self.yn,
        )

    def __hash__(self):
        return hash((self.xlim[0], self.xlim[1], self.xn, self.ylim[0], self.ylim[1], self.yn))

    def __eq__(self, other):
        return self is other or hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def _set_real_space_parameters(self, xlim, ylim, xn, yn, v):
        self.xlim = xlim  # radians
        self.ylim = ylim  # radians
        self.xspan = xlim[1] - xlim[0]  # total side length
        self.yspan = ylim[1] - ylim[0]  # total side length
        self.xn = xn  # number of pixel on each side
        self.yn = yn  # number of pixel on each side
        self.xres = self.xspan / xn  # radians
        self.yres = self.yspan / yn  # radians
        self.fsky = self.xspan * self.yspan / (4 * np.pi)

        # grid point boundry coordinate
        self.xcoord_boundry = np.linspace(xlim[0], xlim[1], xn + 1)
        self.ycoord_boundry = np.linspace(ylim[0], ylim[1], yn + 1)
        # grid point coordinate
        self.xcoord = (self.xcoord_boundry[:-1] + self.xcoord_boundry[1:]) / 2
        self.ycoord = (self.ycoord_boundry[:-1] + self.ycoord_boundry[1:]) / 2
        # grid point 2d coordiante
        self.x, self.y = np.meshgrid(self.xcoord, self.ycoord, indexing="ij")
        self.rmap_shape = (xn, yn)
        # sanity check
        if len(self.xcoord) != self.xn:
            raise ValueError("Length of xcoord does not match xn")

        if len(self.ycoord) != self.yn:
            raise ValueError("Length of ycoord does not match yn")

        if self.rmap_shape != self.x.shape:
            raise ValueError("rmap_shape does not match the shape of x mesh")

        if self.rmap_shape != self.y.shape:
            raise ValueError("rmap_shape does not match the shape of y mesh")

        # there is no reason xres and yres should be different. In particular, if this is not true, then the whole aliasing
        # business becomes more complicated (Different Nyquist frequency in x and y direction).
        # so we are enforcing this from the start.
        if not np.allclose(self.xres, self.yres):
            print(self.xres, self.yres)
            raise ValueError(
                "xres and yres should be equal. if this error is raised during sampling setup, then there could be rounding errors that result in loosing or adding pixels when used with resboost. change padding or resboost until the error goes away."
            )
        return

    def _set_ell_space_parameters(self, v):
        # the ell space conventions follows the galsim convention.
        # C.f. https://github.com/GalSim-developers/GalSim/blob/5a3a428cdff31f71482fc74abb08c255bd7f00fa/galsim/pse.py#L33
        # The y coordinates are truncated by half due to real fft constraint.

        # (the longer coordinate).  # axis 0 of rfft is the normal fft frequency coordinate
        self.lxcoord = 2 * np.pi * fft.fftfreq(self.xn, d=self.xres)
        self.lxcoord = fft.fftshift(self.lxcoord)  # sort the 0 frequency mode to the center of the spectrum
        self.lxn = len(self.lxcoord)
        # (the shorter coordinate). # axis 1 of rfft is half+1 of the normal fft frequency coordinate
        self.lycoord = 2 * np.pi * fft.rfftfreq(self.yn, d=self.yres)
        self.lyn = len(self.lycoord)
        self.lxlim = np.array([np.min(self.lxcoord), np.max(self.lxcoord)])
        self.lylim = np.array([np.min(self.lycoord), np.max(self.lycoord)])
        assert np.allclose(self.lxlim, np.array([self.lxcoord[0], self.lxcoord[-1]]))
        assert np.allclose(self.lylim, np.array([self.lycoord[0], self.lycoord[-1]]))

        self.lxres = np.abs(self.lxcoord[1] - self.lxcoord[0])
        self.lyres = np.abs(self.lycoord[1] - self.lycoord[0])

        # define the Nyquist frequency and check that it is consistent with what we expect
        # the definition is np.pi/self.xres = np.pi/self.yres
        # if we look at the np.fft.fftfreq, we see that it depends on the even/old number of samples
        # if even then np.max(np.abs(self.lxcoord)) is the Nyquist frequency.
        # if odd, then np.max(np.abs(self.lxcoord)) is smaller than the Nyquist frequency, it is Nyquist * (n-1)/n
        self.Nyquist = np.pi / self.xres

        if self.xn % 2 == 0:
            if not np.allclose(self.Nyquist, np.max(np.abs(self.lxcoord))):
                raise ValueError("Nyquist frequency not matching with lxcoord for even xn")
        else:
            if not np.allclose(self.Nyquist * (self.xn - 1) / self.xn, np.max(np.abs(self.lxcoord))):
                raise ValueError("Nyquist frequency not matching with lxcoord for odd xn")

        if self.yn % 2 == 0:
            if not np.allclose(self.Nyquist, np.max(np.abs(self.lycoord))):
                raise ValueError("Nyquist frequency not matching with lycoord for even yn")
        else:
            if not np.allclose(self.Nyquist * (self.yn - 1) / self.yn, np.max(np.abs(self.lycoord))):
                raise ValueError("Nyquist frequency not matching with lycoord for odd yn")

        self.lx, self.ly = np.meshgrid(self.lxcoord, self.lycoord, indexing="ij")
        self.lnrm = np.sqrt(self.lx**2 + self.ly**2)  # a heatmap of the norms of each pixel
        self.lmap_shape = (self.lxn, self.lyn)
        self.XPS_conversion_ell2real2d = (self.xres / self.xn) * (self.yres / self.yn)

        if self.lnrm.shape != self.lmap_shape:
            raise ValueError("Shape of lnrm doesn't match lmap_shape")
        return

    def print_summary(self):
        print(
            f"rsp grid limit   = [{np.degrees(self.xlim[0]):.3f}, {np.degrees(self.xlim[1]):.3f}] x [{np.degrees(self.ylim[0]):.3f}, {np.degrees(self.ylim[1]):.3f}] deg"
        )
        print(f"rsp grid span  = [{np.degrees(self.xspan):.3f}, {np.degrees(self.yspan):.3f}] deg")
        print(f"rsp res        = {np.degrees(self.xres):.8f} {np.degrees(self.yres):.8f} deg")

        print(f"rsp grid shape = {self.rmap_shape}")
        print(f"fsky           = {self.fsky}")

        print(f"lsp grid limit = [{self.lxlim[0]:.3f} {self.lxlim[1]:.3f}] x [{self.lylim[0]:.3f} {self.lylim[1]:.3f}]")
        print(f"lsp grid span  = [{(self.lxlim[1]-self.lxlim[0]):.3f} {(self.lylim[1]-self.lylim[0]):.3f}]")
        print(f"lsp res        = {self.lxres:.8f}, {self.lyres:.8f}")
        print(f"(lxn, lyn)     = {self.lmap_shape}")
        print(f"grid Nyquist   = {self.Nyquist}")
        return

    # plotting
    def plot_rmap(self, rmap, f=None, ax=None, title=""):
        return self.plotter.plot_rmap(rmap, f=f, ax=ax, title=title)

    def plot_lmap(self, lmap, f=None, ax=None, title=""):
        return self.plotter.plot_lmap(lmap, f=f, ax=ax, title=title)

    def plot_array_ij(self, A, indexing="ij"):
        return self.plotter.plot_array(A, indexing=indexing)

    # # kernels
    # kernel_lspunif = staticmethod(kernel_lspunif)
    # kernel_lsppixeltophat = staticmethod(kernel_lsppixeltophat)
    # kernel_lspKS = staticmethod(kernel_lspKS)


class GridAnalysis(Grid):
    """
    All the PS computations are accelerated via JAX JIT.
    """

    def __init__(
        self,
        xlim,
        ylim,
        xn: int,
        yn: int,
        unit: str = "deg",
        bin_centers=None,
        bin_edges=None,
        nlbin: int = 200,
        bin_method: str = "lin",
        bin_center_method: str = "area_avg",
        lmin: None | float = None,
        lmax: None | float = None,
        cl_template: list | np.ndarray | None = None,
        cl_interp_lmin: None | float = None,
        v=False,
        **kwargs,
    ):
        # initialize Grid
        super().__init__(xlim, ylim, xn, yn, unit, v)
        # initialize PSE quantities
        self._setup_PSE(
            bin_centers,
            bin_edges,
            nlbin,
            bin_method,
            bin_center_method,
            lmin,
            lmax,
            cl_template,
            cl_interp_lmin,
            v,
        )

    def _setup_PSE(
        self,
        bin_centers,
        bin_edges,
        nlbin,
        bin_method,
        bin_center_method,
        lmin,
        lmax,
        cl_template,
        cl_interp_lmin,
        v,
    ):
        """
        Add a power spectrum (PS) estimator. This method setup the power spectrum estimator for the current grid setup.
        It calculates the average power in each ell bin using linear or logarithmic bins (specified by 'lin' or 'log').
        The bin center is chosen as the area-averaged center.

        The ell binning is independent of the CG (coarse graining) methods used for sampling.
        The power spectrum is computed starting from lmin, which is approximately equal to the ell resolution,
        and up to lmax, which is approximately the Nyquist frequency.

        Reference: https://github.com/GalSim-developers/GalSim/blob/5a3a428cdff31f71482fc74abb08c255bd7f00fa/galsim/pse.py#L33

        Args:
            method (str, optional): Method for binning the ell norms. Defaults to 'log' (logarithmic) but can be set to 'lin' (linear).
            nlbin (int, optional): Number of bins. Defaults to 10.
            lmin, lmax: By default, these are None (as described above). Alternatively, you can specify the outermost edges for the ell bins.
            v (bool, optional): Defaults to True.

        Added attributes:
            pse_method
            pse_nlbin
            pse_lx: on the full fft grid not the rfft grid, also not fft shifted (i.e. 0 frequency at the beginning)
            pse_ly: on the full fft grid not the rfft grid, also not fft shifted (i.e. 0 frequency at the beginning)
            pse_lnrm: on the full fft grid not the rfft grid

            pse_ledges
            pse_lvals
        """
        # first compute the l-norm grid
        # here we need the coordinate for the full grid, not the real fft coords
        # notice here we do not need to do fft shift
        lxcoord_tmp = 2 * np.pi * fft.fftfreq(self.xn, d=self.xres)
        lycoord_tmp = 2 * np.pi * fft.fftfreq(self.yn, d=self.yres)
        self.pse_lx, self.pse_ly = np.meshgrid(lxcoord_tmp, lycoord_tmp, indexing="ij")
        self.pse_lnrm = np.sqrt(self.pse_lx**2 + self.pse_ly**2)

        # need to specify lmin and lmax if bin_centers or bin_edges are not specified
        if bin_centers is None and bin_edges is None:
            if bin_method == "lin" or bin_method == "lin_2step":
                if lmin is None:
                    lmin = 0
                if lmax is None:
                    lmax = np.max(self.lnrm) + self.lxres
            if bin_method == "log":
                if lmin is None:
                    lmin = np.min([self.lxres, self.lyres])
                if lmax is None:
                    lmax = np.max(self.lnrm) + self.lxres

        ellbin = EllBin(
            bin_centers,
            bin_edges,
            nlbin=nlbin,
            bin_method=bin_method,
            bin_center_method=bin_center_method,
            lmin=lmin,
            lmax=lmax,
            cl_template=cl_template,
            cl_interp_lmin=cl_interp_lmin,
        )
        self.pse_nlbin = ellbin.nlbin
        self.pse_ledges = ellbin.ledges
        self.pse_lvals = ellbin.lvals
        self.pse_lvals_mid = ellbin.lvals_mid
        self._compute_basecount()
        self._add_lnrm_digitized()
        return

    def _add_lnrm_digitized(self):
        ledge_min = np.min(self.pse_ledges)
        ledge_max = np.max(self.pse_ledges)
        pse_lnrm_mask = (self.pse_lnrm.ravel() < ledge_min) | (self.pse_lnrm.ravel() >= ledge_max)
        self.pse_lindex = np.digitize(self.pse_lnrm.ravel(), bins=self.pse_ledges, right=False) - 1
        self.pse_lnrm_digitized = self.pse_lvals[self.pse_lindex]
        self.pse_lnrm_digitized[pse_lnrm_mask] = np.nan
        self.pse_lnrm_digitized = self.pse_lnrm_digitized.reshape(self.pse_lnrm.shape)
        return

    def crop(self, m, pad_size: int | None = None):
        """
        Crop the input maps batch-wise and check for compatibility.

        Parameters
        ----------
        m : ndarray
            Shape (..., xn, yn). e.g., (nsample, ntracer, xn, yn).
        pad_size : int or None, optional
            If not None, remove pad_size number of pixels on each side before computing.

        """
        assert m.ndim >= 2, "input map must have at least 2 dimensions"
        if pad_size == 0:
            pad_size = None
        if pad_size is not None:
            m = m[..., pad_size:-pad_size, pad_size:-pad_size]
        assert m.shape[-2] == self.xn
        assert m.shape[-1] == self.yn
        return m

    def _compute_basecount(self):
        """
        Added attributes:
            _pse_count
        """
        # Compute the binned statistics of pse_lnrm using the bins pse_ledges
        # returns an array over the ell bins of sum_{|ell| in bin} = count
        # We will later use this histogram to compute the power spectrum
        self._pse_count, _ = np.histogram(self.pse_lnrm, self.pse_ledges)
        self._pse_count = np.float64(self._pse_count)
        # if (self._pse_count == 0).any():
        #     print(self.pse_lvals)
        #     print(self._pse_count)
        # raise ValueError('Warning("bin definition resulted in >=1 empty bin!")')
        return

    # regarding the staticargument, see the previous notes on hashing
    # @partial(jit, static_argnums=0)
    def _bin_power1d_uve(self, C):  # unbiased variance estimator
        # normalization given by unbiased variance estimator

        # note here np.histogram flattens pse_lnrm automatically
        P, _ = jnp.histogram(self.pse_lnrm, self.pse_ledges, weights=C)

        # The Cochran correction (this is for std not variance)
        # c4 = 1 - 1 / 4 / self._pse_count - 7 / 32 / self._pse_count**2 - 19 / 128 / self._pse_count**3
        # return P.real / (self._pse_count) / c4 / c4

        # sample variance correction # note there are 2 modes per \vec ell, one real one complex
        weight = 2 * self._pse_count / (2 * self._pse_count - 1)
        return P.real / (self._pse_count) * weight

    # regarding the staticargument, see the previous notes on hashing
    # @partial(jit, static_argnums=0)
    def _bin_power1d(self, C):
        # adapted from Galsim PSE
        # The call to `histogram` just returns an array over the
        # ell bins of sum_{|ell| in bin} C_{ell_x,ell_y}
        # alternatively one can also use scipy.stats.binned_statistic

        # note here np.histogram flattens pse_lnrm automatically
        P, _ = jnp.histogram(self.pse_lnrm, self.pse_ledges, weights=C)

        # we have previous enforeced that (self._pse_count == 0).any() should never happen. However, even
        # in this case, the function will output inf as per numpy convention. pyplot will ignore inf just as nan
        return P.real / (self._pse_count)

    # regarding the staticargument, see the previous notes on hashing
    # @partial(jit, static_argnums=0)
    def PSE_XPS(self, m):
        """
        Compute the cross power spectrum between a set of maps.

        Parameters
        ----------
        m : ndarray
            Shape (ntracer, nx, ny).
        Returns
        -------
        ndarray
            Cross power spectrum of shape (ntracer, ntracer, nbins).
        """
        assert m.ndim == 3
        # m has shape (ntracer, nx, ny)
        ntracer = m.shape[0]

        # batch compute the FT of both maps
        # note, here using fft2 not real-fft2. by using the full ell-space grid, we don't need to manually expand the complex conjugates
        # m_lsp has shape (ntracer, nlx, nly)
        m_lsp = fft.fft2(m)
        m_lsp_conj = jnp.conjugate(m_lsp)

        m1 = jnp.empty((ntracer * ntracer, m_lsp.shape[1], m_lsp.shape[2]), dtype="complex128")
        m2 = jnp.empty((ntracer * ntracer, m_lsp.shape[1], m_lsp.shape[2]), dtype="complex128")
        for i in range(ntracer):
            for j in range(ntracer):
                m1 = m1.at[i * ntracer + j].set(m_lsp[i])
                m2 = m2.at[i * ntracer + j].set(m_lsp_conj[j])

        # see note https://github.com/GalSim-developers/GalSim/blob/5a3a428cdff31f71482fc74abb08c255bd7f00fa/galsim/pse.py#L33
        # Use the internal function above to bin, and account for the normalization of the FFT.
        # Recall that power has units of angle^2, which is the reason why we need a self.dx^2 in the
        # equations below in addition to the standard 1/N^2 coming from the FFTs.
        product = m1 * m2 * self.XPS_conversion_ell2real2d
        XPS = vmap(self._bin_power1d_uve)(product)

        # reshape the i-j spectra
        XPS = jnp.reshape(XPS, (ntracer, ntracer, XPS.shape[1]))
        return XPS

    def PSE_XPSbatched(self, m, pad_size=None, summary=True):
        assert m.ndim == 4
        m = self.crop(m, pad_size)
        XPS = vmap(self.PSE_XPS)(m)
        if summary:
            XPS_mean = jnp.mean(XPS, axis=0)
            XPS_std = jnp.std(XPS, axis=0)
            XPS_low = XPS_mean - XPS_std
            XPS_high = XPS_mean + XPS_std
            return XPS_mean, XPS_low, XPS_high
        else:
            return XPS

    def PSE_XPStheory(self, ell, XPS, kernel=None, kernel_power=2, digitize=False):
        """compute binned theory XPS on the grid
        Parameters
        ----------
        ell: 1d array
            the ell values of the theory XPS
        XPS: 3d array
            the theory XPS, shape (ntracer,ntracer,nell)
        kernel: 2d array
            the kernel to be applied to the theory XPS
        kernel_power: int
            the power of the kernel to be applied to the theory XPS
        digitize: bool
            if False, then the theory XPS is interpolated onto the grid. it is binned when measured per usual PSE.
            if True, then for each ring defined by ledges, the theory XPS is injected at the corr. lvals.
        """

        assert XPS.ndim == 3
        assert XPS.shape[0] == XPS.shape[1]
        assert XPS.shape[2] == len(ell)

        ntracer = XPS.shape[0]
        result = []
        for i in range(ntracer):
            tmp = []
            for j in range(ntracer):
                # the variance map m*conj(m)
                if digitize:
                    lnrm = self.pse_lnrm_digitized
                else:
                    lnrm = self.pse_lnrm
                m2 = self._distribute_radial_f(lnrm, ell, XPS[i, j])
                # if there is kernel, then multiply by the square of the kernel. (^2 since we are dealing with variance)
                if kernel is not None:
                    assert kernel.shape == self.lnrm.shape
                    assert isinstance(kernel_power, int)
                    m2 *= kernel ** (kernel_power)
                tmp.append(self._bin_power1d(m2))
            result.append(tmp)
        return jnp.array(result)

    def _distribute_radial_f(self, r_query, r, f_r):
        """
        compute a radial function f_r = f(r) onto a single grid.
        r_query is a 2d array that has the value of r at each grid point.
        r and f_r are both 1d array.
        """
        assert r_query.ndim == 2
        shape = r_query.shape
        res = jnp.interp(r_query.ravel(), r, f_r)
        return res.reshape(shape)

    # def PSE_k2g(self, k):  # can be moved to the main grid class
    #     """
    #     kappa -> gamma_1, gamma_2 via Kaiser-Squires.
    #     No padding, be careful with boundry conditions.

    #     Parameters
    #     ----------
    #     k : ndarray
    #         Shape (nx, ny). a real space kappa map.
    #     """

    #     kernel_KS1, kernel_KS2 = self.kernel_lspKS(self.lx, self.ly)
    #     # self.plot_lmap(lmap=KS1, f=None, ax=None, title=r"$KS_1$")
    #     # self.plot_lmap(lmap=KS2, f=None, ax=None, title=r"$KS_2$")

    #     k = fft.fftshift(fft.rfft2(k), axes=(0,))
    #     # self.plot_lmap(lmap=np.abs(kell), f=None, ax=None, title=r"$\kappa$")

    #     g1 = kernel_KS1 * k
    #     g2 = kernel_KS2 * k
    #     # self.plot_lmap(lmap=np.abs(g1), f=None, ax=None, title=r"$|\gamma_1|$")
    #     # self.plot_lmap(lmap=np.abs(g2), f=None, ax=None, title=r"$|\gamma_2|$")

    #     g1 = fft.irfft2(fft.ifftshift(g1, axes=(0,)))
    #     g2 = fft.irfft2(fft.ifftshift(g2, axes=(0,)))
    #     # self.plot_rmap(rmap=g1, f=None, ax=None, title=r"$\gamma_1$")
    #     # self.plot_rmap(rmap=g2, f=None, ax=None, title=r"$\gamma_2$")
    #     return g1, g2

    @partial(jit, static_argnums=0)
    def _ell_vals(self, m, ell_range):
        mask = (ell_range[0] <= self.pse_lnrm.ravel()) & (self.pse_lnrm.ravel() <= ell_range[1])
        return jnp.where(mask, m.ravel(), np.nan)

    def ell_vals(self, m, ell_range, pad_size=None):
        """
        Parameters
        ----------
        m : ndarray
            Shape (nmaps, nx, ny).
        ell_range :
            (ell_min, ell_max)
        """
        m = self.crop(m, pad_size)
        m_fft = fft.fft2(m)
        print(m_fft.shape)
        values = vmap(self._ell_vals, in_axes=(0, None))(m_fft, ell_range)
        # drop nan
        values = values.ravel()
        values = values[~np.isnan(values)]
        values = np.real(values)
        print(values.shape)
        return values


class EllBin:
    """define different ways to set up ell bins. This class is used in the GridAnalysis and the GridSampling classes."""

    def __init__(
        self,
        bin_centers=None,
        bin_edges=None,
        nlbin: int = 200,
        bin_method: str = "lin",
        bin_center_method: str = "area_avg",
        lmin: None | float = None,
        lmax: None | float = None,
        cl_template: list | np.ndarray | None = None,
        cl_interp_lmin: None | float = None,
    ):
        """setup ell bin edges and centers
        Parameters
        ----------
        lmin: float
            for sampling, when bin_method='lin' we recommand 0
            for log bins, we recommand np.min([self.lxres, self.lyres])
        lmax: float
            for sampling, we recommand np.max(self.lnrm) + self.lxres
        bin_method: str
            'lin' or 'log'
        bin_center_method: str
            'center', 'area_avg', 'cl_interp'
        nlbin: int

        Attributes
        ----------
        ledges: 1d array
            bin edges
        lvals: 1d array
            bin centers
        lvals_mid: 1d array
            bin centers, mid points
        """
        if bin_centers is not None and bin_edges is not None:
            self.ledges = bin_edges
            self.lvals = bin_centers
            self.nlbin = len(bin_centers)
        else:
            # bin edges
            assert bin_method in ["lin", "lin_2step", "log"]
            assert bin_center_method in ["center", "area_avg", "cl_interp"]
            assert lmin is not None or lmax is not None
            self.bin_method = bin_method
            self.bin_center_method = bin_center_method
            self.nlbin = int(nlbin)
            self.lmin = lmin
            self.lmax = lmax

            if bin_method == "lin":
                self.set_bin_edges_lin()
            elif bin_method == "lin_2step":
                self.set_bin_edges_lin_2()
            elif bin_method == "log":
                self.set_bin_edges_log()
            # bin centers
            if bin_center_method == "center":
                self.set_bin_centers_center()
            elif bin_center_method == "area_avg":
                self.set_bin_centers_area_avg()
            elif bin_center_method == "cl_interp":
                assert bin_method in ["lin", "lin_2step"], "bin_method must be 'lin' or 'lin_2step' if bin_center_method is 'cl_interp'"
                assert cl_template is not None and cl_interp_lmin is not None
                self.set_bin_centers_cl_interp(cl_template, cl_interp_lmin)

        self.lvals_mid = (self.ledges[1:] + self.ledges[:-1]) / 2

        return

    def set_bin_edges_lin(self):
        self.ledges = np.linspace(self.lmin, self.lmax, self.nlbin + 1)
        return

    def set_bin_edges_lin_2(self):
        # linear bins with 2 resolutions
        # n_1000 bins given to ell in [lmin,1000]
        # the rest of nlbin is given to ell > 1000
        n_1000 = 50
        assert self.nlbin >= n_1000 + 20  # 20 is for margins
        ledges1 = np.linspace(self.lmin, 1000, n_1000)
        ledges2 = np.linspace(1000, self.lmax, self.nlbin - n_1000 + 2)
        self.ledges = np.append(ledges1, ledges2[1:])
        return

    def set_bin_edges_log(self):
        assert self.lmin > 0, "lmin must be > 0 if bin_method is 'log'"
        self.ledges = np.logspace(np.log10(self.lmin), np.log10(self.lmax), self.nlbin + 1)
        return

    def set_bin_centers_center(self):
        self.lvals = (self.ledges[1:] + self.ledges[:-1]) / 2
        return

    def set_bin_centers_area_avg(self):
        # area averaged: integral_l0^l1 (l*2*pi*l*dl) / (pi*(l1^2-l0^2))
        self.lvals = (2.0 / 3.0) * (self.ledges[1:] ** 3 - self.ledges[:-1] ** 3) / (self.ledges[1:] ** 2 - self.ledges[:-1] ** 2)
        return

    def set_bin_centers_cl_interp(self, cl_template, cl_interp_lmin):
        """
        compute linear ell bins, and define the effective centers "l*" as C(l*) = <C(l)>_l0^l1(over the ring)
        Parameters
        ----------
        cl_template : 2d array
            first element is ell, second element is C_ell
        cl_interp_lmin : None or float
            l<cl_interp_lmin are set to 0 when calculating the effective bin centers
        """
        assert cl_template is not None, "cl_template must be provided if bin_sep_method is 'interp'"
        assert cl_template.shape[0] == 2, "cl_template[0] must be ell, cl_template[1] must be C_ell"
        assert cl_interp_lmin is not None, "cl_interp_lmin must be provided if bin_sep_method is 'interp'"

        def _calc_closest_x(x, y, y_target):
            # find the x value in an arrray such that y(x*) is closest to y_target
            # in case of multiple occurrences of the minimum values, the indices corresponding to the first occurrence are returned.
            idx = (np.abs(y - y_target)).argmin()
            x_star = x[idx]
            return x_star

        lvals = []
        for i in range(self.nlbin):
            l0 = self.ledges[i]
            l1 = self.ledges[i + 1]
            l_in_interval = np.linspace(l0, l1, 100)
            cl_in_interval = np.interp(l_in_interval, cl_template[0], cl_template[1])
            cl_in_interval[l_in_interval <= cl_interp_lmin] = 0
            # <C(l)>_l0^l1(over the ring) = (int_l0^l1 c(l) 2 pi l dl) / (pi (l1^2-l0^2))
            cl_avg = np.trapz(cl_in_interval * 2 * np.pi * l_in_interval, l_in_interval)
            cl_avg /= np.pi * (l1**2 - l0**2)
            # now find the l* that satisfies C(l*) = <C(l)>_l0^l1
            lvals.append(_calc_closest_x(l_in_interval, cl_in_interval, cl_avg))

        self.lvals = np.array(lvals)
        return


# plot maps that are defined on the grid with 2-d coordinates x and y (as per meshgrid "ij" indexing convention)
class PlotGrid:
    def __init__(self, x, y, lx, ly):
        """plot maps that are defined on the grid.
        Parameters
        ----------
        x: numpy.ndarray
            2-d coordinates in radians, "ij" indexing convention (see numpy.meshgrid)
        y: numpy.ndarray
            2-d coordinates in radians, "ij" indexing convention (see numpy.meshgrid)
        lx: numpy.ndarray
            2-d coordinates in ell space, rfft convention, fftshifted
        ly: numpy.ndarray
            2-d coordinates in ell space, rfft convention
        """
        assert x.ndim == 2 and y.ndim == 2, "x and y must be 2-d coordinates in radians"
        assert lx.ndim == 2 and ly.ndim == 2, "lx and ly must be 2-d coordinates in ell space"
        self.x = x
        self.y = y
        self.lx = lx
        self.ly = ly
        return

    def plot_rmap(self, rmap, f=None, ax=None, title=""):
        if ax is None and f is None:
            f, ax = plt.subplots()
            show_flag = True
        else:
            show_flag = False
        x, y = np.degrees(self.x), np.degrees(self.y)
        c = ax.pcolormesh(x, y, rmap, cmap="RdBu_r", vmin=rmap.min(), vmax=rmap.max())
        f.colorbar(c, ax=ax)
        ax.axis([x.min(), x.max(), y.min(), y.max()])
        ax.set_xlabel(r"x [deg]")
        ax.set_ylabel(r"y [deg]")
        ax.set_aspect(1)
        ax.set_title(title)
        if show_flag:
            plt.show()
        return c

    def plot_lmap(self, lmap, f=None, ax=None, title=""):
        # the input map and coord are the (unshifted) fourier space map
        if ax is None and f is None:
            f, ax = plt.subplots()
            show_flag = True
        else:
            show_flag = False
        # lmap = fft.fftshift(lmap, axes=0)
        # lx, ly = fft.fftshift(self.lx, axes=0), fft.fftshift(self.ly, axes=0)
        c = ax.pcolormesh(self.lx, self.ly, lmap, cmap="RdBu_r", vmin=lmap.min(), vmax=lmap.max())
        f.colorbar(c, ax=ax)
        ax.axis([self.lx.min(), self.lx.max(), self.ly.min(), self.ly.max()])
        ax.set_xlabel(r"$\ell_x$")
        ax.set_ylabel(r"$\ell_y$")
        ax.set_aspect(1)
        ax.set_title(title)
        if show_flag:
            plt.show()
        return c

    @staticmethod
    def plot_array(A, indexing="ij"):
        """
        Visualize A in ij indexing fashion.
        Plots a 2D array (matrix) 'A' with 'ij' indexing where the [0,0]
        element is in the lower left corner.

        Specifically:
        - The [0,0] element is in the lower left corner of the plot.
        - The [0,1] element is on the top of the [0,0] element.
        - The [1,0] element is to the right of the [0,0] element.

        Parameters:
        A (numpy.ndarray): A 2D array to be plotted.
        """
        if indexing == "ij":
            plt.imshow(A.T, origin="lower", cmap="RdBu_r")
            plt.colorbar(label="Value")
            plt.xlabel("j index")
            plt.ylabel("i index")
            plt.show()
        else:
            raise NotImplementedError("currently only support 'ij' indexing")
        return
    
def kernel_lspunif(lmap_shape):
    """
    Uniform kernel in ell space (same shape as grid, i.e., up to Nyquist frequency).

    Parameters:
    -----------
    lmap_shape : tuple
        The shape of the lmap grid.

    Returns:
    --------
    ndarray
        The uniform kernel in ell space.

    """
    kernel = jnp.ones(lmap_shape)
    return kernel


def kernel_lsppixeltophat(xres, yres, lx, ly):
    """
    Compute the Fourier transform of the top hat window function (in real space)
    with the window size equal to the real space pixel size.

    Parameters:
    -----------
    xres : float
        The grid size in the x-direction in radians.
    yres : float
        The grid size in the y-direction in radians.
    lx : ndarray
        The lx coordinates with the shape of the lmap grid.
    ly : ndarray
        The ly coordinates with the shape of the lmap grid.

    Returns:
    --------
    ndarray
        The Fourier transform of the top hat window function.

    Notes:
    ------
    The real space top hat window is the product of two top hat windows in the x and y directions.

    """
    kernel = jnp.sinc(xres * lx / 2 / jnp.pi) * np.sinc(yres * ly / 2 / jnp.pi)
    return kernel


def kernel_lspKS(lx, ly, check=False):
    # operators that convert kappa_1,2 to shear_1,2 maps, note, here we implicitely assumed the grid coordiantes follow from g.x and g.y
    # meaning, kx,ky = np.meshgrid( kxcoord,  kycoord, indexing='ij')
    # this convention is also followed by kuindex
    """
    the two outputs are the KS operator that converts a kappa to gamma_1 and gamma_2 in ell space
    the output contains two versions

    KS1,2: same shape as lmap_shape
    """

    denom = lx**2 + ly**2

    KS1 = np.divide((lx**2 - ly**2), denom, out=np.zeros(denom.shape, dtype=float), where=denom != 0)

    KS2 = np.divide((2 * lx * ly), denom, out=np.zeros(denom.shape, dtype=float), where=denom != 0)

    if check:
        numeric_offset = 1e-13
        KS1_old = (lx**2 - ly**2) / (denom + numeric_offset)
        KS2_old = (2 * lx * ly) / (denom + numeric_offset)
        assert np.allclose(KS1_old, KS1)
        assert np.allclose(KS2_old, KS2)
        # print("KS op passed accuracy checks")

    return KS1, KS2

