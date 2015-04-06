from __future__ import division
import pytest
from ad import adnumber
import numpy as np
from adarray import array, adapply, adsum, addot
import random

def check_gradient(ad_arr, arr_of_ad, vars):
    grad1 = ad_arr.gradient(vars)

    grad2 = []
    for v in vars:
        curr_d = np.zeros(ad_arr.shape)
        for i,xi in np.ndenumerate(arr_of_ad):
            curr_d[i] = xi.d(v)
        grad2.append(curr_d)

    assert grad1 == grad2


def test_simple_polynomials():
    x,y = adnumber(np.random.normal(size=2))

    num_dims = 5
    polys = []
    for _ in range(3):
        polys.append([(x ** random.randint(1,5)) * (y ** random.randint(1,5)) for _ in range(num_dims)])
    u,v,w = polys

    z1 = adsum(array(u) * array(v) * array(w))
    z2 = np.sum(np.array(u) * np.array(v) * np.array(w))

    grad1 = z1.gradient([x,y])
    grad2 = z2.gradient([x,y])

    assert grad1 == grad2
    
    hess1 = z1.hessian([x,y])
    hess2 = z2.hessian([x,y])

    assert hess1 == hess2


def test_simple_addot():
    x,y = adnumber(np.random.normal(size=2))

    num_dims = 5
    polys = []
    for _ in range(2):
        polys.append([(x ** random.randint(1,5)) * (y ** random.randint(1,5)) for _ in range(num_dims)])
    u,v = polys

    z1 = addot(array(u) , array(v))
    z2 = np.dot(np.array(u) , np.array(v))

    grad1 = z1.gradient([x,y])
    grad2 = z2.gradient([x,y])

    assert grad1 == grad2
    
    hess1 = z1.hessian([x,y])
    hess2 = z2.hessian([x,y])

    assert hess1 == hess2

def test_simple_addot2():
    x,y = adnumber(np.random.normal(size=2))

    left_dims = 5
    right_dims = 10
    polys = []
    for _ in range(left_dims+1):
        polys.append([(x ** random.randint(1,5)) * (y ** random.randint(1,5)) for _ in range(right_dims)])
    b = polys[0]
    A = polys[1:]

    #A = np.array(A).transpose()

    z2 = np.dot(np.array(A) , np.array(b))
    z1 = addot(array(A) , array(b))

    grad1 = z1.gradient([x,y])
    grad2 = [np.array([z2i.d(u) for z2i in z2]) for u in x,y]

    #print grad1, "\n", grad2
    assert np.all(np.array(grad1) == np.array(grad2))
    
    hess1 = [z1.d2c(i,j) for i in x,y for j in x,y if i is not j]
    hess1 += [z1.d2(i) for i in x,y]
    hess2 = [np.array([z2i.d2c(i,j) for z2i in z2]) for i in x,y for j in x,y if i is not j]
    hess2 += [np.array([z2i.d2(i) for z2i in z2]) for i in x,y]

    print hess1, "\n", hess2
    assert np.all(np.array(hess1) == np.array(hess2))
