from __future__ import division
import pytest
from ad import adnumber
import numpy as np
from adarray import array, adapply, adsum
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


def test_simple():
    x,y = adnumber(np.random.normal(size=2))

    polys = []
    for _ in range(2):
        polys.append([(x ** random.randint(1,5)) * (y ** random.randint(1,5)) for _ in range(2)])
    u,v = polys

    z1 = adsum(array(u) * array(v))
    z2 = np.sum((np.array(u) * np.array(v)))

    grad1 = z1.gradient([x,y])
    grad2 = z2.gradient([x,y])

    assert grad1 == grad2
