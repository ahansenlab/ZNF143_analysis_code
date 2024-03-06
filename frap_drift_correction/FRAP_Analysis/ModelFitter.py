"""
ModelFitter.py - Fits FRAP recovery models
"""

# numerical tools
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import i0, i1, iv, kv
from .finiteNucleusUtilities import preprocessFM, fitFunFM

# INVLAP
from .invlap import invlap

K_OFF_1_MAX = 0.1
DIFF_COEFF_MAX = 50

class FRAPModel:
    """
    Parent class for FRAP models
    """

    def __init__(self, image_reader, start_frame=None):
        self.ImageReader = image_reader
        self.x_data = self.ImageReader.get_frame_metadata()[:, 1]
        self.y_data = self.ImageReader.get_mean_intensity_data()
        self.w = self.ImageReader.get_real_roi_radius()
        print(self.w)

        # initialize some parameters that may be estimated
        self.D = -1.0

        if start_frame is None:
            self.start_frame = np.argmin(self.y_data)
        else:
            self.start_frame = start_frame
        self.popt = None
        self.pcov = None

    @staticmethod
    def func(x, *args):
        return None

    def fit(self, p0=None):
        """
        Fit model using nonlinear curve fitting

        :return: optimal parameter fits
        """
        self.popt, self.pcov = curve_fit(self.func,
                                         self.x_data[self.start_frame:],
                                         self.y_data[self.start_frame:], p0)
        return self.popt, self.pcov

    def get_parameters(self):
        """
        Getter for optimal parameters

        :return: A list of optimal parameters
        """
        return self.popt

    def get_cov(self):
        """
        Getter for covariance matrix

        :return: A list of variances
        """
        return self.popt

    def get_D(self):
        """
        Getter for diffusion coefficient

        :return: A diffusion coefficient
        """
        return self.D

    def get_fit_pts(self):
        """
        Gets model values run at all time points

        :return: Timepoints and model values at those timepoints
        """
        return (self.x_data[self.start_frame:],
               self.func(self.x_data[self.start_frame:], *self.popt))

    def make_plt_plot(self):
        """
        Low level plotting method to display
        :return:
        """
        if self.popt is None:
            raise TypeError("popt is None. Model is not fit")

        fig, ax = plt.subplots()

        # plot real data
        ax.plot(self.x_data, self.y_data, 'b-', label="Raw data")

        # plot fit
        ax.plot(self.x_data[self.start_frame-1:],
                self.func(self.x_data[self.start_frame-1:], *self.popt),
                'r-', label="Model fit")

        # set plot aesthetic parameters
        ax.set_title('Model Fit')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mean Intensity')
        ax.legend()
        plt.show()


class BasicExponential(FRAPModel):
    """
    Basic exponential model for FRAP recovery
    """

    def __init__(self, image_reader, start_frame=None):
        super().__init__(image_reader, start_frame)

    @staticmethod
    def func(x, a, tau):
        return a * (1 - np.exp(-tau * x))

class PureDiffusion(FRAPModel):
    """
    Pure diffusive recovery model assuming
    circular bleaching as derived by SOUMPASIS 1983
    https://www-ncbi-nlm-nih-gov.libproxy.mit.edu/pmc/articles/PMC1329018/pdf/biophysj00223-0097.pdf
    """
    def __init__(self, image_reader, start_frame=None):
        super().__init__(image_reader, start_frame)

    @staticmethod
    def func(x, t_d):
        return np.nan_to_num(np.exp(-(2 * t_d) / x ) * (i0((2 * t_d) / x) + i1((2 * t_d) / x)))

    def fit(self, p0=None):
        """
        Fit model using nonlinear curve fitting

        :return: optimal parameter fits
        """
        self.popt, self.pcov = curve_fit(self.func,
                                         self.x_data[self.start_frame:],
                                         self.y_data[self.start_frame:], p0)
        self.D = (self.w ** 2) / (4 * self.popt[0])
        return self.popt, self.pcov

    def get_parameters(self):
        """
        Getter for optimal parameters

        :return: A list of optimal parameters
        """
        return [self.popt[0], self.D]

class OneReaction(FRAPModel):
    """
    Reaction dominant recovery model assuming
    diffusion is much much faster than reaction according to
    https://www-ncbi-nlm-nih-gov.libproxy.mit.edu/pmc/articles/PMC1304253/#fd5
    """
    def __init__(self, image_reader, start_frame=None):
        super().__init__(image_reader, start_frame)

    @staticmethod
    def func(x, C_eq, k_off):
        return 1 - C_eq * np.exp(- k_off * x)

    def fit(self, p0=None):
        """
        Fit model using nonlinear curve fitting

        :return: optimal parameter fits
        """
        self.popt, self.pcov = curve_fit(self.func,
                                         self.x_data[self.start_frame:],
                                         self.y_data[self.start_frame:], p0)
        self.k_on_app_1 = (self.popt[0] * self.popt[1]) / (1 - self.popt[0])
        return self.popt, self.pcov

    def get_parameters(self):
        """
        Getter for optimal parameters

        :return: A list of optimal parameters
        """
        return [self.popt[1], self.k_on_app_1]


class TwoReaction(FRAPModel):
    """
    Reaction dominant recovery model assuming
    diffusion is much much faster than reaction according to with 2 binding states
    https://www-ncbi-nlm-nih-gov.libproxy.mit.edu/pmc/articles/PMC1304253/#fd5
    """
    def __init__(self, image_reader, start_frame=None):
        super().__init__(image_reader, start_frame)

    @staticmethod
    def func(x, k_1_off, k_2_off, k_1_on, k_2_on):
        C_1_eq = 1 / (1 + (k_1_off/k_1_on) * (1 + (k_2_on / k_2_off)))
        C_2_eq = 1 / (1 + (k_2_off / k_2_on) * (1 + (k_1_on / k_1_off)))
        return 1 - C_1_eq * np.exp(- k_1_off * x) - C_2_eq * np.exp(- k_2_off * x)

    def fit(self, p0=None):
        """
        Fit model using nonlinear curve fitting

        :return: optimal parameter fits
        """
        self.popt, self.pcov = curve_fit(self.func,
                                         self.x_data[self.start_frame:],
                                         self.y_data[self.start_frame:], p0,
                                         bounds=(np.zeros(4), np.full(4, np.inf)))

        return self.popt, self.pcov

class FullOneReactionAverage(FRAPModel):
    """
    Full reaction diffusion solution with a single binding reaction and fitting to
    the average spot intensity.
    https://www-ncbi-nlm-nih-gov.libproxy.mit.edu/pmc/articles/PMC1304253/#fd5
    """
    def __init__(self, image_reader, start_frame=None):
        super().__init__(image_reader, start_frame)

    @staticmethod
    def func(x, k_on, k_off, D_f, w):
        F_eq = k_off / (k_on + k_off)
        C_eq = k_on / (k_on + k_off)
        q = lambda p: np.sqrt((p / D_f) * (1 + k_on / (p + k_off)))
        frap_p = lambda p: (1 / p) - (F_eq / p) * (1 - 2 * kv(1.0, q(p) * w) * iv(1.0, q(p) * w)) * \
                           (1 + k_on / (p + k_off)) - C_eq / (p + k_off)

        x[x==0] += np.finfo(float).eps
        output = invlap(frap_p, x, M=50)
        output[np.isnan(output)] = 0.0
        print('*')
        return output


    def fit(self, p0=None):
        """
        Fit model using nonlinear curve fitting

        :return: optimal parameter fits
        """

        fit_fun = lambda x, k_on, k_off, D_f: self.func(x, k_on, k_off, D_f, self.w)
        lower = np.full(3, np.finfo(float).eps)
        upper = np.full(3, np.inf)
        upper[1] = K_OFF_1_MAX
        upper[2] = DIFF_COEFF_MAX

        self.popt, self.pcov = curve_fit(fit_fun,
                                         self.x_data[self.start_frame:],
                                         self.y_data[self.start_frame:], p0,
                                         bounds=(lower, upper))

        return self.popt, self.pcov

    def get_fit_pts(self):
        """
        Gets model values run at all time points

        :return: Timepoints and model values at those timepoints
        """
        fit_fun = lambda x, k_on, k_off, D_f: self.func(x, k_on, k_off, D_f, self.w)
        return (self.x_data[self.start_frame:],
                fit_fun(self.x_data[self.start_frame:], *self.popt))

class FullTwoReactionAverage(FRAPModel):
    """
    Full reaction diffusion solution with two binding reactions and fitting to
    the average spot intensity.
    https://www-ncbi-nlm-nih-gov.libproxy.mit.edu/pmc/articles/PMC1304253/#fd5
    """
    def __init__(self, image_reader, start_frame=None):
        super().__init__(image_reader, start_frame)

    @staticmethod
    def func(x, k_on_1, k_on_2, k_off_1, k_off_2, D_f, w):
        F_eq = 1 / (1 + k_on_1 / k_off_1 + k_on_2 / k_off_2)
        C_1_eq = 1 / (1 + (k_off_1 / k_on_1) * (1 + (k_on_2 / k_off_2)))
        C_2_eq = 1 / (1 + (k_off_2 / k_on_2) * (1 + (k_on_1 / k_off_1)))
        q = lambda p: np.sqrt((p / D_f) * (1 + k_on_1 / (p + k_off_1) + k_on_2 / (p+ k_off_2)))
        frap_p = lambda p: (1 / p) - (F_eq / p) * (1 - 2 * kv(1.0, q(p) * w) * iv(1.0, q(p) * w)) * \
                           (1 + k_on_1 / (p + k_off_1) + k_on_2 / (p+ k_off_2)) - C_1_eq / (p + k_off_1) - \
                           C_2_eq / (p + k_off_2)

        x[x == 0] += np.finfo(float).eps
        output = invlap(frap_p, x, M = 50)
        output[np.isnan(output)] = 0.0
        print('*')
        return output

    def fit(self, p0=None):
        """
        Fit model using nonlinear curve fitting

        :return: optimal parameter fits
        """

        fit_fun = lambda x, k_on_1, k_on_2, k_off_1, k_off_2, D_f: \
            self.func(x, k_on_1, k_on_2, k_off_1, k_off_2, D_f, self.w)
        lower = np.full(5, np.finfo(float).eps)
        upper = np.full(5, np.inf)
        upper[2] = K_OFF_1_MAX
        upper[4] = DIFF_COEFF_MAX

        self.popt, self.pcov = curve_fit(fit_fun,
                                         self.x_data[self.start_frame:],
                                         self.y_data[self.start_frame:], p0,
                                         bounds=(lower, upper))

        return self.popt, self.pcov

    def get_fit_pts(self):
        """
        Gets model values run at all time points

        :return: Timepoints and model values at those timepoints
        """
        fit_fun = lambda x, k_on_1, k_on_2, k_off_1, k_off_2, D_f: \
            self.func(x, k_on_1, k_on_2, k_off_1, k_off_2, D_f, self.w)
        return (self.x_data[self.start_frame:],
                fit_fun(self.x_data[self.start_frame:], *self.popt))

class FiniteOneReactionAverage(FRAPModel):
    """
    Full reaction diffusion solution with one binding reactions and fitting to
    the average spot intensity with finite nucleus
    https://pubmed.ncbi.nlm.nih.gov/18199661/
    """
    def __init__(self, image_reader, start_frame=None):
        super().__init__(image_reader, start_frame)

    @staticmethod
    def func(x, k_on_1, k_on_2, k_off_1, k_off_2, D_f, w):
        F_eq = 1 / (1 + k_on_1 / k_off_1 + k_on_2 / k_off_2)
        C_1_eq = 1 / (1 + (k_off_1 / k_on_1) * (1 + (k_on_2 / k_off_2)))
        C_2_eq = 1 / (1 + (k_off_2 / k_on_2) * (1 + (k_on_1 / k_off_1)))
        q = lambda p: np.sqrt((p / D_f) * (1 + k_on_1 / (p + k_off_1) + k_on_2 / (p+ k_off_2)))
        frap_p = lambda p: (1 / p) - (F_eq / p) * (1 - 2 * kv(1.0, q(p) * w) * iv(1.0, q(p) * w)) * \
                           (1 + k_on_1 / (p + k_off_1) + k_on_2 / (p+ k_off_2)) - C_1_eq / (p + k_off_1) - \
                           C_2_eq / (p + k_off_2)

        x[x == 0] += np.finfo(float).eps
        output = invlap(frap_p, x, M = 50)
        output[np.isnan(output)] = 0.0
        print('*')
        return output

    def fit(self, p0=None):
        """
        Fit model using nonlinear curve fitting

        :return: optimal parameter fits
        """

        fit_fun = lambda x, k_on_1, k_on_2, k_off_1, k_off_2, D_f: \
            self.func(x, k_on_1, k_on_2, k_off_1, k_off_2, D_f, self.w)
        lower = np.full(5, np.finfo(float).eps)
        upper = np.full(5, np.inf)
        upper[2] = K_OFF_1_MAX
        upper[4] = DIFF_COEFF_MAX

        self.popt, self.pcov = curve_fit(fit_fun,
                                         self.x_data[self.start_frame:],
                                         self.y_data[self.start_frame:], p0,
                                         bounds=(lower, upper))

        return self.popt, self.pcov

    def get_fit_pts(self):
        """
        Gets model values run at all time points

        :return: Timepoints and model values at those timepoints
        """
        fit_fun = lambda x, k_on_1, k_on_2, k_off_1, k_off_2, D_f: \
            self.func(x, k_on_1, k_on_2, k_off_1, k_off_2, D_f, self.w)
        return (self.x_data[self.start_frame:],
                fit_fun(self.x_data[self.start_frame:], *self.popt))
