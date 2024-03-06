"""
finiteNucleusUtilities.py

This file contains helper functions for the finite nucleus model based on the derivation in Mueller et al.,
Evidence for a common mode of transcription factor interaction with chromatin as revealed by improved quantitative FRAP,
Biophysical Journal, 2007. Equation numbers refer to this publication.

This was originally implemented in MATLAB by Florian Mueller, NIH/NIC,
Version 1.0, January 2008 - muellerf@mail.nih.gov

The implementation here was written by Domenic Narducci, MIT, domenicn@mit.edu
"""
import numpy as np
import scipy.special as sc

def preprocessFM(RN, RM, RC, theta, sigma=0):
    """
    This function computes parts of the solution which are independent of fitting parameters
    :param RN: radius of the nucleus [um]
    :param RM: radius of measurement [um]
    :param RC: radius of uniform bleaching [um]
    :param theta: bleach depth
    :param sigma: width of the gaussian distribution (uniform if 0)
    :return: (tuple of series) J1wxi, alpha2, Z
    """
    # TODO: Go back and rename variables
    # For uniform bleaching, include 500 members of bessel series expansion
    if sigma == 0:
        NNN = 5000

    j_zeros = np.array(sc.jn_zeros(1, NNN-1))
    # compute alpha
    alpha2 = np.divide(j_zeros, RN) ** 2
    alpha2 = np.concatenate(([0], alpha2))

    # calculate coefficients Zk of the Bessel expansion
    # Note that the coefficients are not multiplied by Feq.This
    # multiplication is done in the function to calculate the actual FRAP curve
    # This is necessary since Feq depends on the binding rates and the pre - processing is only
    # done for calculations which are independent of the binding rates.

    # Uniform bleaching(Eq.(S.15) in Mueller et al.)
    if sigma == 0:
        J1RC = sc.j1(j_zeros * RC / RN)
        J0 = sc.j0(j_zeros)
        J02 = J0 ** 2

        Z0 = 1 + (theta - 1) * (RC / RN) ** 2
        Z = (theta - 1) * (2 * RC / RN) * (J1RC / j_zeros) / J02
        Z = np.concatenate(([Z0], Z))

    # Spatial averaging of the Bessel - function for the FRAP curve (Eq.(S.17))
    J1w = sc.j1(j_zeros * (RM / RN))
    J1wxi = np.concatenate(([1],  2 * (RN / RM) * J1w / j_zeros))

    return J1wxi, alpha2, Z

def fitFunFM(t, kon, koff, Df, J1wxi, alpha2, Z):

    """
    Computes reaction diffusion circle FRAP solution.
    This function is based on the derivation in Mueller et al., Evidence
    for a common mode of transcription factor interaction with chromatin
    as revealed by improved quantitative FRAP, Biophysical Journal. Equation numbers refer to this publication.
    :param t: Time points to evaluate at
    :param kon: on rate
    :param koff: off rate
    :param Df: diffusion coefficient
    :param J1wxi:
    :param alpha2:
    :param Z:
    :return:
    :return: (series) Reaction diffusion eq solution
    """
    # Calculation of FRAP curve
    eps1 = np.finfo(float).eps # To avoid division by zero and assuring right limiting process
    Feq = (koff + eps1) / (kon + koff + eps1)

    # - Multiply Z with Feq(Compare Eqs.(S.12) and (S.15) in Mueller et al.)
    # This step is performed here since Feq depends on the binding rates and
    # thus can not be calcuated in the pre - processing step.
    Z = Feq * Z

    # - Calculate exponential decay rates(Eq.(S.4))
    ww = 0.5 * (Df * alpha2 + kon + koff)
    vv = np.sqrt(ww**2 - koff * Df * alpha2)

    bet = ww + vv
    gam = ww - vv

    ea = np.exp(np.outer(-bet,t))
    eb = np.exp(np.outer(-gam,t))

    # - Calculate coeffiecients of series expansion
    UU = -(0.5 / koff) * (-ww - vv + koff) * (ww - vv) / vv # Eq.(S.11)
    VV = (0.5 / koff) * (-ww + vv + koff) * (ww + vv) / vv # Eq.(S.11)

    U = UU * Z # Eq.(S.11)
    V = VV * Z # Eq.(S.11)

    W = kon * U / (-bet + koff) # Eq.(S.10)
    X = kon * V / (-gam + koff + eps1) # Eq.(S.10)

    # - Calculate FRAP curve
    frap = (((U + W) * J1wxi)@ea+((V+X)*J1wxi) @ eb) # Eq.(S.16)
    return frap


# physical_size = 0.05857254468880701 #mm
# img_size = 512
# radius = 15.0
# start_frame = 25
# w = 1000 * radius * (physical_size / img_size)
# RN = 10
# theta = 0.45
# preprocess = preprocessFM(RN, w, w, theta)
# fit_fun = lambda x, k_on, k_off, D_f: fitFunFM(x, k_on, k_off, D_f, *preprocess)
# print(fit_fun([1,2,3], 0.4, 0.5, 1.0))