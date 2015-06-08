import size_history
import util
import numpy as np
from scipy.misc import comb as binom
import random

def test_functional_sfs_matches_constant():
    tau = random.expovariate(1.)
    N = 1000. * random.random()
    n_max = 100
    hist1 = size_history.ConstantTruncatedSizeHistory(n_max, tau, N)
    hist2 = size_history.FunctionalTruncatedSizeHistory(n_max, tau, lambda t: 1. / N)
    assert np.allclose(hist1.etjj, hist2.etjj)
    assert np.allclose(hist1.scaled_time, hist2.scaled_time)

def test_functional_sfs_matches_exponential():
    tau = .1
    N_bottom = 1.
    N_top = .1
    n_max = 100
    hist1 = size_history.ExponentialTruncatedSizeHistory(n_max, tau, N_top, N_bottom)
    hist2 = size_history.FunctionalTruncatedSizeHistory(n_max, tau, lambda t: 1. / (N_bottom*np.exp(-hist1.growth_rate * t)))
    assert np.allclose(hist1.etjj, hist2.etjj)
    assert np.allclose(hist1.scaled_time, hist2.scaled_time)

def test_functional_sfs_matches_piecewise():
    tau1, tau2 = .1,.2
    N1,N2 = 1.,.1
    n_max = 100
    hist1 = size_history.PiecewiseHistory([size_history.ExponentialTruncatedSizeHistory(n_max, tau1, N2,N1),
                                           size_history.ConstantTruncatedSizeHistory(n_max, tau2, N2)])
    hist2 = size_history.FunctionalTruncatedSizeHistory(n_max, tau1+tau2,
                                                        lambda t: 1. / (N1*np.exp(-hist1.pieces[0].growth_rate * t)) if t < tau1 else 1. / N2)
    assert np.allclose(hist1.etjj, hist2.etjj)
    assert np.allclose(hist1.scaled_time, hist2.scaled_time)
    
