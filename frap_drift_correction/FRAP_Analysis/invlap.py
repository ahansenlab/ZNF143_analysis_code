"""
invlap.py
Computes inverse laplace transform. Implemented according
to the code from invlap.m. Modified de Hoog algorithm
"""

import numpy as np

def invlap(F, t, alpha = 0, tol = 1e-9, M=20, P = tuple()):
    """
    Implementation of de Hoog quotient difference method
    with accelerated convergence for continued fraction expansion
    Time vector is split in segments of equal magnitude which are
    inverted individually giving better overall accuracy

    Attribution: Karl Hollenbeck. Department of Hydrodynamics and
    Water Resources Technical University of Denmark
    :param F: laplace-space function of the form F(s, *P) where s is the
    Laplace parameter and return column vector as result
    :param t: column vector of times for which function values are
    sought
    :param alpha: largest pole of F (zero by default)
    :param tol: numerical tolerance of approaching pole (default 1e-9)
    :param P: optional parameters to be passed to F
    :return: vector of real-space values f(t)
    """
    # split up t vector in pieces of same order of magnitude,
    # invert one piece at a time. Simultaneous inversion for times covering
    # several orders of magnitudes gives inaccurate results for the small times.
    f = np.array([])
    allt = t
    logallt = np.log10(t)
    iminlogallt = int(np.floor(np.min(logallt)))
    imaxlogallt = int(np.ceil(np.max(logallt)))
    for ilogt in np.arange(iminlogallt, imaxlogallt + 1):
        t = allt[np.logical_and(logallt >= ilogt, logallt < (ilogt + 1))]
        if t.size != 0: # maybe no elements in that magnitude
            T = max(t) * 2
            gamma = (alpha - np.log(tol)) / (2 * T)
            # NOTE: The correction alpha -> alpha - log(tol) / (2 * T) is not in de
            # Hoog's paper, but in Mathematica's Mathsource (NLapInv.m) implementation of
            # inverse transforms
            nt = len(t)
            run = np.arange(0, 2 * M + 1) # so there are 2M + 1 terms in Fourier series expansion

            # find F argument call F with it get 'a' coefficients in power series
            s = gamma + 1j * np.pi * run / T
            a = F(s, *P)
            a[0] = a[0] / 2 # zero term is halved

            # build up e and q tables. superscript is now row index, subscript column
            e = np.zeros((2*M + 1, M + 1), dtype = 'complex_')
            q = np.zeros((2*M    , M + 1), dtype = 'complex_')
            e[:, 0] = np.zeros(2*M + 1, dtype = 'complex_')
            q[:, 1] = np.divide(a[1:2*M + 1], a[0:2 * M], dtype = 'complex_')

            for r in np.arange(1, M + 1): # step through columns (called r...)
                e[0: 2 * (M - r) + 1, r] = q[1: 2 * (M - r) + 2, r] \
                                               - q[0: 2 * (M - r) + 1, r] \
                                               + e[1: 2 * (M - r) + 2, r - 1]
                if r < M: # one column fewer for q
                    rq = r + 1
                    q[0: 2 * (M - rq) + 2, rq] = np.multiply(q[1: 2 * (M - rq) + 3, rq - 1],
                                                                 e[1: 2 * (M - rq) + 3, rq - 1], dtype='complex_') \
                                                     / e[0: 2 * (M - rq) + 2, rq - 1]

            # build up d vector (index shift: 1)
            d = np.zeros((2 * M + 1, 1), dtype = 'complex_')
            d[0, 0] = a[0]
            d[1:2 * M:2, 0] = -q[0, 1: M + 1].T # these 2 lines changed after niclas
            d[2:2 * M + 1:2, 0] = -e[0, 1: M + 1].T

            # build up A and B vectors(index shift: 2)
            # - now make into matrices, one row for each time
            A = np.zeros((2 * M + 2, nt), dtype = 'complex_')
            B = np.zeros((2 * M + 2, nt), dtype = 'complex_')
            A[1,:] = d[0, 0] * np.ones((1, nt))
            B[0: 2,:] = np.ones((2, nt))
            z = np.exp(1j * np.pi * t.T/T) # row vector
            # after niclas back to the paper(not: z = exp(-i * pi * t / T)) !!!
            for n in np.arange(2, 2 * M + 2):
                A[n,:] = A[n - 1,:] + d[n - 1, 0] * np.ones((1, nt)) * z * A[n - 2,:] # different index
                B[n,:] = B[n - 1,:] + d[n - 1, 0] * np.ones((1, nt)) * z * B[n - 2,:] # shift for d!

            # double acceleration
            h2M = .5 * (np.ones((1, nt)) + (d[2 * M - 1, 0] - d[2 * M, 0]) * np.ones((1, nt)) * z)
            R2Mz = -h2M * (np.ones((1, nt))
                           - np.sqrt(np.ones((1, nt)) + d[2 * M, 0] * np.ones((1, nt)) * z / (h2M)**2))
            A[2 * M + 1,:] = A[2 * M,:] + R2Mz * A[2 * M - 1,:]
            B[2 * M + 1,:] = B[2 * M,:] + R2Mz * B[2 * M - 1,:]

            # inversion, vectorized for times, make result a column vector
            fpiece = ((1 / T) * np.exp(gamma * t.T) * np.real(A[2*M+1,:] / B[2*M+1,:]) ).T
            f =np.concatenate((f, fpiece)) # put pieces together

    return f

# test = lambda s: np.divide(1, (s + 1), dtype = 'complex_')
#
# print(invlap(test, np.arange(1, 50)))
# print(np.exp(-np.arange(1, 50) ))
# print(invlap(test, np.arange(1, 50)) - np.exp(-np.arange(1, 50) ))