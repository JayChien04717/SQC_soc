from __future__ import annotations
import numpy as np
import warnings
from scipy.optimize import least_squares
from inspect import signature
from copy import deepcopy



def complex_fit(f, xdata, ydata, p0=None, weights=None, **kwargs):
    """
    Wrapper around scipy least_square for complex functions

    Args:
        f: The model function, f(x, …). It must take the independent variable
            as the first argument and the parameters to fit as separate
            remaining arguments.
        xdata: The independent variable where the data is measured.
        ydata: The dependent data.
        p0: Initial guess on independent variables.
        weights: Optional weighting in the calculation of the cost function.
        kwargs: passed to the leas_square function

    Returns:
        A tuple with the optimal parameters and the covariance matrix
    """

    if (np.array(ydata).size - len(p0)) <= 0:
        raise ValueError(
            "yData length should be greater than the number of parameters."
        )

    def residuals(params, x, y):
        """ Computes the residual for the least square algorithm"""
        if weights is not None:
            diff = weights * f(x, *params) - y
        else:
            diff = f(x, *params) - y
        flat_diff = np.zeros(diff.size * 2, dtype=np.float64)
        flat_diff[0 : flat_diff.size : 2] = diff.real
        flat_diff[1 : flat_diff.size : 2] = diff.imag
        return flat_diff

    kwargs_ls = kwargs.copy()
    kwargs_ls.setdefault("max_nfev", 1000)
    kwargs_ls.setdefault("ftol", 1e-2)
    opt_res = least_squares(residuals, p0, args=(xdata, ydata), **kwargs_ls)

    jac = opt_res.jac
    cost = opt_res.cost

    pcov = np.linalg.inv(jac.T.dot(jac))
    pcov *= cost / (np.array(ydata).size - len(p0))

    popt = opt_res.x

    return popt, pcov


def guess_edelay_from_gradient(freq, signal, n=-1):

    dtheta = np.mean(np.angle(signal[-n:] / zeros2eps(signal[:n])))
    df = np.mean(np.diff(freq))

    return dtheta / df / 2 / np.pi


def smooth_gradient(signal):
    def dnormaldx(x, x_0, sigma):
        return -(x - x_0) * np.exp(-0.5 * ((x - x_0) / sigma) ** 2)

    conv_kernel_size = max(min(100, signal.size // 20), 2)

    conv_kernel = dnormaldx(
        x=np.arange(0.5, conv_kernel_size + 0.5, 1),
        x_0=conv_kernel_size / 2,
        sigma=conv_kernel_size / 8,
    )

    gradient = np.convolve(signal, conv_kernel, "same")
    gradient[: conv_kernel_size // 2] = gradient[
        conv_kernel_size // 2 : 2 * (conv_kernel_size // 2)
    ][::-1]
    gradient[-(conv_kernel_size // 2) :] = gradient[
        -2 * (conv_kernel_size // 2) : -(conv_kernel_size // 2)
    ][::-1]

    return gradient


eps = np.finfo(float).eps


def zeros2eps(x):

    """
    args:
        x: float, complex, or numpy array
    
    return:
        y: numpy array

    replace the zeros of a float or numpy array bien the smallest float number
    """

    y = np.array(x)
    y[np.abs(y) < eps] = eps

    return y


def dB(x):

    return 20 * np.log10(np.abs(x))


def deg(x):

    return np.angle(x) * 180 / np.pi


def get_prefix(x):

    prefix = [
        "y",  # yocto
        "z",  # zepto
        "a",  # atto
        "f",  # femto
        "p",  # pico
        "n",  # nano
        "u",  # micro
        "m",  # mili
        "",
        "k",  # kilo
        "M",  # mega
        "G",  # giga
        "T",  # tera
        "P",  # peta
        "E",  # exa
        "Z",  # zetta
        "Y",  # yotta
    ]

    max_x = np.abs(np.max(x))

    if max_x > 10 * eps:

        index = int(np.log10(max_x) / 3 + 8)
        return (x * 10 ** (-3 * (index - 8)), prefix[index])

    else:

        return (0, "")


def get_prefix_str(x, precision=2):

    return "%.{}f %s".format(precision) % get_prefix(x)





def transmission(freq, f_0, kappa):

    num = 1
    den = 2j * (freq - f_0) + kappa

    return num / zeros2eps(den)


def reflection(freq, f_0, kappa, kappa_c_real, phi_0=0):

    num = 2j * (freq - f_0) + kappa - 2*kappa_c_real*(1+1j*np.tan(phi_0))
    den = 2j * (freq - f_0) + kappa

    return num / zeros2eps(den)


def reflection_mismatched(freq, f_0, kappa, kappa_c_real, phi_0):

    return reflection(freq, f_0, kappa, kappa_c_real, phi_0)


def hanger(freq, f_0, kappa, kappa_c_real, phi_0=0):

    num = 2j * (freq - f_0) + kappa - kappa_c_real*(1+1j*np.tan(phi_0))
    den = 2j * (freq - f_0) + kappa

    return num / zeros2eps(den)


def hanger_mismatched(freq, f_0, kappa, kappa_c_real, phi_0):

    return hanger(freq, f_0, kappa, kappa_c_real, phi_0)

resonator_dict = {
    "transmission": transmission,
    "t": transmission,
    "reflection": reflection,
    "r": reflection,
    "reflection_mismatched": reflection_mismatched,
    "rm": reflection_mismatched,
    "hanger": hanger,
    "h": hanger,
    "hanger_mismatched": hanger_mismatched,
    "hm": hanger_mismatched,
}

def get_fit_function(geometry, amplitude=True, edelay=True):
    if type(geometry) == str:
        resonator_func = resonator_dict[geometry]
    else:
        resonator_func = geometry

    if not amplitude and not edelay:
        return resonator_func

    elif amplitude and not edelay:
        def fit_func(*args):
            return resonator_func(*args[:-2]) * (args[-2] + 1j * args[-1])

        return fit_func

    elif not amplitude and edelay:
        def fit_func(*args):
            return resonator_func(*args[:-1]) * np.exp(2j * np.pi * args[-1] * args[0])

        return fit_func

    elif amplitude and edelay:
        def fit_func(*args):
            return (
                resonator_func(*args[:-3])
                * (args[-3] + 1j * args[-2])
                * np.exp(2j * np.pi * args[-1] * args[0])
            )

        return fit_func

    else:

        raise Exception("Unreachable")

class ResonatorParams(object):
    def __init__(self, params, geometry, freq = None, signal = None):

        self.resonator_func = resonator_dict[geometry]
        self.params = params

        self.freq = freq
        self.signal = signal

        if self.resonator_func == transmission:
            self.f_0_index = 0
            self.kappa_index = 1
            if len(self.params) in [4, 5]:
                self.re_a_in_index = 2
                self.im_a_in_index = 3
            if len(self.params) in [3, 5]:
                self.edelay_index = -1

        if self.resonator_func in [reflection, hanger]:
            self.f_0_index = 0
            self.kappa_index = 1
            self.kappa_c_real_index = 2
            if len(self.params) in [5, 6]:
                self.re_a_in_index = 3
                self.im_a_in_index = 4
            if len(self.params) in [4, 6]:
                self.edelay_index = -1

        if self.resonator_func in [reflection_mismatched, hanger_mismatched]:
            self.f_0_index = 0
            self.kappa_index = 1
            self.kappa_c_real_index = 2
            self.phi_0_index = 3
            if len(self.params) in [6, 7]:
                self.re_a_in_index = 3
                self.im_a_in_index = 4
            if len(self.params) in [5, 7]:
                self.edelay_index = -1

    def tolist(self):
        return np.array(self.params)

    @property
    def f_0(self):
        if hasattr(self, "f_0_index"):
            return self.params[self.f_0_index]
        else:
            return None

    @property
    def kappa(self):
        if hasattr(self, "kappa_index"):
            return self.params[self.kappa_index]
        else:
            None

    @property
    def kappa_i(self):
        if hasattr(self, "kappa_index") and hasattr(self, "kappa_c_real_index"):
            return self.params[self.kappa_index] - self.params[self.kappa_c_real_index]
        else:
            return None

    @property
    def kappa_c_real(self):
        if hasattr(self, "kappa_c_real_index"):
            return self.params[self.kappa_c_real_index]
        else:
            return None

    @property
    def kappa_c(self):
        return self.kappa_c_real

    @property
    def a_in(self):
        if hasattr(self, "re_a_in_index") and hasattr(self, "im_a_in_index"):
            return (
                self.params[self.re_a_in_index] + 1j * self.params[self.im_a_in_index]
            )
        else:
            return None

    @property
    def re_a_in(self):
        a_in = self.a_in
        if a_in is not None:
            return np.real(a_in)
        else:
            return None

    @property
    def im_a_in(self):
        a_in = self.a_in
        if a_in is not None:
            return np.imag(a_in)
        else:
            return None

    @property
    def edelay(self):
        if hasattr(self, "edelay_index"):
            return self.params[self.edelay_index]
        else:
            return None

    @property
    def phi_0(self):
        if hasattr(self, "phi_0_index"):
            return self.params[self.phi_0_index]
        else:
            return None

    def str(self, latex=False, separator=", ", precision=2, only_f_and_kappa=False, f_precision=2, red_warning = False):
        kappa = {False: "kappa/2pi", True: r"$\kappa/2\pi$"}
        kappa_i = {False: "kappa_i/2pi", True: r"$\kappa_i/2\pi$"}
        kappa_c = {False: "kappa_c/2pi", True: r"$\kappa_c/2\pi$"}
        f_0 = {False: "f_0", True: r"$f_0$"}
        phi_0 = {False: "phi_0", True: r"$\varphi_0$"}

        if self.edelay is not None:
            edelay_str = "%sedelay = %ss" % (separator, get_prefix_str(self.edelay, precision))
        else:
            edelay_str = ""

        if self.resonator_func == transmission:
            kappa_str = r"%s%s = %sHz" % (
                separator,
                kappa[latex],
                get_prefix_str(self.kappa, precision),
            )
        else:
            kappa_str = r"%s%s = %sHz%s%s = %sHz" % (
                separator,
                kappa[latex],
                get_prefix_str(self.kappa, precision),
                separator,
                kappa_c[latex],
                get_prefix_str(self.kappa_c, precision),
            )

        if self.resonator_func in [hanger_mismatched, reflection_mismatched]:
            if red_warning and self.phi_0 is not None and np.abs(self.phi_0) > 0.25:
                phi_0_str = r"%s = %0.2f rad" % (phi_0[latex], self.phi_0)
                phi_0_str = "%s/!\\ "%separator + phi_0_str + " /!\\"
            else:
                phi_0_str = r"%s%s = %0.2f rad" % (separator, phi_0[latex], self.phi_0)
        else:
            phi_0_str = ""

        f_0_str = r"%s = %sHz" % (f_0[latex], get_prefix_str(self.f_0, f_precision))

        if only_f_and_kappa:
            return f_0_str + kappa_str
        else:
            return f_0_str + kappa_str + phi_0_str + edelay_str

    def __str__(self) -> str:
        return self.str()

    def __repr__(self):
        return self.str()

    def __call__(self, freq, *args, **kwargs):
        amplitude = self.a_in is not None
        edelay = self.edelay is not None

        fit_func = get_fit_function(self.resonator_func, amplitude, edelay)

        if len(args) == 0 and len(kwargs) == 0:
            params = self.params
        else:
            if len(kwargs) == 0:
                params = np.copy(self.params)
                params[:len(args)] = args
            else:
                resonator = deepcopy(self)
                for key in kwargs:
                    resonator.params[resonator.__dict__[key + "_index"]] = kwargs[key]
                resonator.params[:len(args)] = args
                params = resonator.params
        
        return fit_func(freq, *params)
    
    def plot(
            self,
            fig = None,
            plot_not_corrected = True,
            font_size = None,
            plot_circle = True,
            center_freq = False,
            only_f_and_kappa = False,
            precision = 2,
            alpha_fit = 1.0,
            style = 'Normal',
            title = None,
            params = None,
        ):
        plot(
            self.freq,
            self.signal,
            self(self.freq),
            fig=fig,
            fit_params=self,
            params=params,
            plot_not_corrected=plot_not_corrected,
            font_size=font_size,
            plot_circle=plot_circle,
            center_freq=center_freq,
            only_f_and_kappa=only_f_and_kappa,
            precision=precision,
            alpha_fit=alpha_fit,
            style=style,
            title=title
        )






def get_abcd(freq, signal, rec_depth=0):

    freq_center = np.mean(freq)

    x_design = np.ones((2, freq.size))
    x_design[1, :] = freq - freq_center

    signal_grad = smooth_gradient(signal)

    for _ in range(rec_depth + 1):

        xx = (x_design * np.abs(signal_grad)) @ x_design.T
        xdyx = (signal * x_design * np.abs(signal_grad)) @ x_design.T
        xdycx = (np.conj(signal) * x_design * np.abs(signal_grad)) @ x_design.T
        xdy2x = (np.abs(signal) ** 2 * x_design * np.abs(signal_grad)) @ x_design.T

        up_right = np.linalg.inv(xx) @ xdyx
        bottom_left = np.linalg.inv(xdy2x) @ xdycx

        to_diag = np.zeros((4, 4), dtype=complex)
        to_diag[:2, 2:] = up_right
        to_diag[2:, :2] = bottom_left

        v, w = np.linalg.eig(to_diag)

        abcd = w[:, np.argmin(np.abs(1 - v))]

        signal_grad = (abcd[2] + abcd[3] * (freq - freq_center)) ** -2

    abcd[0::2] -= abcd[1::2] * freq_center

    fit = (abcd[0] + abcd[1] * freq) / (abcd[2] + abcd[3] * freq)

    return abcd, fit


def abcd2params(abcd, geometry):

    a, b, c, d = abcd

    if resonator_dict[geometry] == reflection:

        f_0 = -0.5 * np.real(a / b + c / d)
        a_in = b / d

        kappa_c_real = np.real(1j * (c / d - a / b))
        kappa_i = -np.imag(a / b + c / d)
        kappa = kappa_i + kappa_c_real

        return f_0, kappa, kappa_c_real, np.real(a_in), np.imag(a_in)

    if resonator_dict[geometry] == reflection_mismatched:

        f_0 = -np.real(c / d)
        a_in = b / d

        kappa_c_imag = np.real(a / b - c / d)
        kappa_c_real = np.real(1j * (c / d - a / b))
        kappa_i = -np.imag(a / b + c / d)

        kappa = kappa_i + kappa_c_real

        phi_0 = np.angle(kappa_c_real - 1j*kappa_c_imag)

        return f_0, kappa, kappa_c_real, phi_0, np.real(a_in), np.imag(a_in)

    elif resonator_dict[geometry] == transmission:

        signal_f_0_before = (a - b * np.real(c / d)) / (c - d * np.real(c / d))
        a, b = a - c * b / d, 0
        signal_f_0_after = (a - b * np.real(c / d)) / (c - d * np.real(c / d))
        a = a * signal_f_0_before / signal_f_0_after

        kappa = -2 * np.imag(c / d)
        f_0 = -np.real(c / d)
        a_in = 2j * a / d

        return f_0, kappa, np.real(a_in), np.imag(a_in)

    elif resonator_dict[geometry] == hanger:

        kappa_c_r = np.real(1j * (c / d - a / b))
        kappa_i_r = -np.imag(a / b + c / d)
        f_0 = -0.5 * np.real(a / b + c / d)
        a_in = b / d

        kappa_c_real = 2 * kappa_c_r
        kappa_i = kappa_i_r - kappa_c_r

        kappa = kappa_i + kappa_c_real

        return f_0, kappa, kappa_c_real, np.real(a_in), np.imag(a_in)

    elif resonator_dict[geometry] == hanger_mismatched:

        f_0 = -np.real(c / d)
        a_in = b / d

        kappa_c_imag = 2*np.real(a / b - c / d)
        kappa_c_real = 2*np.real(1j * (c / d - a / b))
        kappa_i = -2 * np.imag(c / d) - kappa_c_real

        kappa = kappa_i + kappa_c_real

        phi_0 = np.angle(kappa_c_real - 1j*kappa_c_imag)

        return f_0, kappa, kappa_c_real, phi_0, np.real(a_in), np.imag(a_in)


def meta_fit_edelay(freq, signal, rec_depth=0):

    quick_fit = get_abcd

    guess_edelay = guess_edelay_from_gradient(freq, signal)

    edelay_span = 1.5 / (np.max(freq) - np.min(freq))

    edelay_array = guess_edelay + np.linspace(-1, 1, 1001) * edelay_span
    l2_error_array = np.zeros_like(edelay_array)

    for i, ed in enumerate(edelay_array):

        s = signal * np.exp(-2j * np.pi * freq * ed)
        _, abcd_fit = quick_fit(freq, s, rec_depth)

        l2_error_array[i] = np.sum(np.abs(s - abcd_fit) ** 2) / freq.size

    return edelay_array[np.argmin(l2_error_array)]


def fit_signal(
    freq,
    signal,
    geometry,
    fit_amplitude=True,
    fit_edelay=True,
    final_ls_opti=True,
    allow_mismatch=True,
    rec_depth=1,
    api_warning=True,
):
    if api_warning:
        warnings.warn("fit_signal() is deprecated, please use analyze() instead, and analyze().plot() to display data.", UserWarning)

    if fit_edelay:
        edelay = meta_fit_edelay(freq, signal, rec_depth)
    else:
        edelay = 0

    if resonator_dict[geometry] == reflection and allow_mismatch:
        geometry = "rm"
    elif resonator_dict[geometry] == hanger and allow_mismatch:
        geometry = "hm"

    corrected_signal = signal * np.exp(-2j * np.pi * edelay * freq)

    quick_fit = get_abcd

    abcd, _ = quick_fit(freq, corrected_signal, rec_depth)

    params = abcd2params(abcd, geometry)
    if not fit_amplitude:
        params = params[:-2]
    if fit_edelay:
        params = [*params, edelay]

    fit_func = get_fit_function(geometry, fit_amplitude, fit_edelay)

    if final_ls_opti:
        params, _ = complex_fit(fit_func, freq, signal, params)
    
    resonator_params = ResonatorParams(params, geometry, freq, signal)

    if resonator_params.phi_0 is not None and  np.abs(resonator_params.phi_0) > 0.25:
        warnings.warn("Extracted phi_0 greater than 0.25, this might indicate a big impedance mismatch, values of kappa_i and kappa_c might be affected, you can try to set: allow_mismatch=False", UserWarning)

    return fit_func, ResonatorParams(params, geometry, freq, signal)

def analyze(
    freq,
    signal,
    geometry,
    fit_amplitude=True,
    fit_edelay=True,
    final_ls_opti=True,
    allow_mismatch=True,
    rec_depth=1,
):
    return fit_signal(
        freq,
        signal,
        geometry,
        fit_amplitude,
        fit_edelay,
        final_ls_opti,
        allow_mismatch,
        rec_depth,
        api_warning = False)[1]
