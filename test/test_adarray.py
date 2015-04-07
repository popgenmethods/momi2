from __future__ import division
import pytest
from ad import adnumber, ADF
import numpy as np
from adarray import array, adapply, dot, tensordot, fftconvolve, sum
import random


def check_gradient(x_ad, x_np, *vars):
    grad1 = x_ad.gradient(vars)

    grad2 = []
    for v in vars:
        curr = np.zeros(shape=x_np.shape)
        for i,xi in np.ndenumerate(x_np):
            curr[i] = xi.d(v)
        grad2.append(curr)

    grad1,grad2 = [np.array(g) for g in grad1,grad2]
    assert np.max(np.abs(np.log(grad1 / grad2))) < 1e-12

    hess1 = x_ad.hessian(vars)
    hess2 = []
    for v in vars:
        currRow1,currRow2 = [],[]
        #hess1.append(currRow1)
        hess2.append(currRow2)

        for u in vars:
            # make d2 entry for hess2
            curr = np.zeros(shape=x_np.shape)
            for i,xi in np.ndenumerate(x_np):
                if u is v:
                    curr[i] = xi.d2(v)
                else:
                    curr[i] = xi.d2c(u,v)
            currRow2.append(curr)

    hess1,hess2 = [np.array(h) for h in hess1,hess2]
    assert np.max(np.abs(np.log(hess1 / hess2))) < 1e-12


def get_random_monomials(args, shape=None, maxorder=5, minorder=1):
    ret = np.zeros(shape=shape) + adnumber(1.0)

    for x in args:
        assert isinstance(x, ADF)
        for i,_ in np.ndenumerate(ret):
            xpow = x ** random.randint(minorder,maxorder)
            ret[i] = xpow * ret[i]
    return ret
            

def test_polynomial():
    x,y = adnumber(np.random.normal(size=2))

    u,v,w = [get_random_monomials([x,y], shape=5) for _ in range(3)]

    z1 = sum(array(u) * array(v) * array(w))
    z2 = np.sum(u * v * w)

    check_gradient(z1,z2,x,y)


def test_dot():
    x,y = adnumber(np.random.normal(size=2))

    u,v = [get_random_monomials([x,y], shape=5) for _ in range(2)]

    z1 = dot(array(u) , array(v))
    z2 = np.dot(u , v)

    check_gradient(z1,z2,x,y)

def test_dot2():
    x,y = adnumber(np.random.normal(size=2))

    u = get_random_monomials([x,y], shape=(10,5))
    v = get_random_monomials([x,y], shape=5)

    z1 = dot(array(u) , array(v))
    z2 = np.dot(u , v)

    check_gradient(z1,z2,x,y)

def test_convolve():
    x,y = adnumber(np.random.normal(size=2))
    
    u = get_random_monomials([x,y], shape=(10,))
    v = get_random_monomials([x,y], shape=(10,))

    z1 = fftconvolve(array(u), array(v))
    z2 = np.convolve(u,v)

    check_gradient(z1,z2,x,y)

def test_getitem():
    x,y = adnumber(np.random.normal(size=2))

    u = get_random_monomials([x,y], shape=(5,5))
    v = get_random_monomials([x,y], shape=5)

    z2 = u *  v[:,None]
    z1 = array(u) * (array(v)[:,None])

    check_gradient(z1,z2,x,y)

def test_tensordot():
    x,y,z = adnumber(np.random.normal(size=3))
    
    u = get_random_monomials([x,y,z], shape=(2,3,4,5))
    v = get_random_monomials([x,y,z], shape=(2,3,4,5))

    axes = [[2,3]]*2

    z2 = np.tensordot(u,v, axes=axes)
    z1 = tensordot(array(u), array(v), axes=axes)

    check_gradient(z1,z2,x,y,z)
