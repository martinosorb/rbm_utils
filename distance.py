import numpy as np
from scipy.special import expit


def propup(data, w, hb):
    return expit(np.dot(data, w) + hb)


def dRBM(rbm, data1, data2):
    """
    Computes the distance defined in "Blindfold learning of an accurate neural
    metric" (Gardella, Marre, Mora; arXiv 2017), page 4.

    Inputs:
    rbm -- a Theano RBM object fitted to the joint dataset of data1 and data2
    data1 -- the first dataset (visible activations)
    data2 -- the second dataset (visible activations)
    """
    w = rbm.W.get_value()
    hbias = rbm.hbias.get_value()
    assert data1.shape[1] == rbm.n_visible
    assert data2.shape[1] == rbm.n_visible
    h1 = np.mean(propup(data1, w, hbias), axis=0)
    h2 = np.mean(propup(data2, w, hbias), axis=0)
    Δh = h2 - h1
    assert len(h1) == rbm.n_hidden
    all_data = np.vstack([data1, data2])
    covm = np.cov(all_data, rowvar=0)
    Wh = w @ Δh
    return np.dot(Wh, covm @ Wh)
