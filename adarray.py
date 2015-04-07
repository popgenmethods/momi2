import ad
from ad import adnumber, ADF, ADV
import numpy as np
from numbers import Number
import scipy, scipy.signal

''' This file hacks the ad module so that we can have ad(np.ndarray) objects'''

'''function analogous to np.array, except with derivatives'''
def array(x):
    if isinstance(x, ADF):
        return adapply(np.array, x)

    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    if np.issubdtype(x.dtype, np.number):
        return x

    # get the variables to differentiate against
    adentries = []
    for i,xi in np.ndenumerate(x):
        if isinstance(xi, ADF):
            adentries.append((i,xi))
        elif not isinstance(xi,Number):
            raise TypeError(str((i,xi)))
    variables = _get_variables([xi for _,xi in adentries])

    # initialize the dictionaries of derivatives
    lc_dict, qc_dict, cp_dict = {}, {}, {}
    d_dicts = (lc_dict, qc_dict, cp_dict)

    # fill the dictionaries of derivatives
    for i,xi in adentries:
        for xi_d, x_d in zip((xi._lc, xi._qc, xi._cp), d_dicts):
            for k in xi_d:
                if k not in x_d:
                    x_d[k] = np.zeros(x.shape)
                x_d[k][i] = xi_d[k]

    x_old = x
    x = np.zeros(x.shape)
    for i,xi in np.ndenumerate(x_old):
        if isinstance(xi,ADF):
            x[i] = xi.x
        elif isinstance(xi, Number):
            x[i] = xi
        else:
            raise Exception

    return ADF(x, lc_dict, qc_dict, cp_dict)


''' for a list of ADF, gets all the directions with derivatives'''
def _get_variables(adfuncs):
    variables = set()
    for x in adfuncs:
        variables |= set(x._lc)
    return variables

'''add shape and length to ADF'''
@property
def shape(self):
    return self.x.shape

def adlen(self):
    return self.x.__len__()

ADF.shape = shape
ADF.__len__ = adlen

'''redefine to_auto_diff to treat numpy arrays as constants'''
def to_auto_diff(x):
    if isinstance(x, ADF):
        return x

    if isinstance(x, Number) or (isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number)):
        # constants have no derivatives to define:
        return ADF(x, {}, {}, {})

    raise NotImplementedError(
        'Automatic differentiation not yet supported for {:} objects'.format(
        type(x))
        )

ad.to_auto_diff = to_auto_diff

''' apply f to x and all its derivatives '''
def adapply(f, x, *args, **kwargs):
    ret_x = f(x.x, *args, **kwargs)
    lc, qc, cp = {},{},{}
    for ret_d, x_d in zip((lc, qc, cp), (x._lc, x._qc, x._cp)):
        for v in x_d:
            ret_d[v] = f(x_d[v], *args, **kwargs)
    return ADF(ret_x, lc, qc, cp)

''' implement __getitem__ for ADF'''
def ad_getitem(self, *args, **kwargs):
    return adapply(np.ndarray.__getitem__, self, *args, **kwargs)

ADF.__getitem__ = ad_getitem

''' implement __setitem__ for ADF'''
def ad_setitem(self, key, value):
    if not isinstance(key, Number):
        raise NotImplementedError
    self.x[key] = value
    for derivatives in (self._lc, self._qc, self._cp):
        for direction in derivatives:
            derivatives[direction][key] = 0.0

ADF.__setitem__ = ad_setitem

''' implement sum for ADF'''
def sum(x, *args, **kwargs):
    if isinstance(x, ADF):
        return adapply(np.sum, x, *args, **kwargs)
    return np.sum(x, *args, **kwargs)

def adarray_sum(self, *args, **kwargs):
    return adapply(np.ndarray.sum, self, *args, **kwargs)

ADF.sum = adarray_sum

''' implements product rule for multiplication-like operations, e.g. matrix/tensor multiplication, convolution'''
def adproduct(prod):
    def f(a,b, *args, **kwargs):
        a,b = to_auto_diff(a), to_auto_diff(b)
        x = prod(a.x,b.x, *args, **kwargs)

        variables = _get_variables([a,b])
        lc, qc, cp = {}, {}, {}
        for i,v in enumerate(variables):
            lc[v] = prod(a.d(v), b.x, *args, **kwargs) + prod(a.x, b.d(v),*args,**kwargs)
            qc[v] = prod(a.d2(v), b.x, *args, **kwargs ) + 2 * prod(a.d(v), b.d(v), *args, **kwargs) + prod(a.x, b.d2(v), *args, **kwargs)

            for j,u in enumerate(variables):
                if i < j:
                    cp[(v,u)] = prod(a.d2c(u,v), b.x, *args, **kwargs) + prod(a.d(u), b.d(v), *args, **kwargs) + prod(a.d(v) , b.d(u), *args, **kwargs) + prod(a.x, b.d2c(u,v), *args, **kwargs)
        return ADF(x, lc, qc, cp)
    return f

'''matrix multiplication, tensor multiplication, and convolution (Fourier domain multiplication)'''
dot = adproduct(np.dot)
tensordot = adproduct(np.tensordot)
fftconvolve = adproduct(scipy.signal.fftconvolve)


''' hack d, d2, and d2c, to work with ad(np.ndarray) types'''
def zero(x):
    try:
        return np.zeros(shape=x.shape)
    except AttributeError:
        return 0.0

def d(self, x=None):
    if x is not None:
        if isinstance(x, ADF):
            try:
                tmp = self._lc[x]
            except KeyError:
                #tmp = 0.0
                tmp = zero(self)
            return tmp
            #return tmp if tmp.imag else tmp.real
        else:
            return 0.0
    else:
        return self._lc

def d2(self, x=None):
    if x is not None:
        if isinstance(x, ADF):
            try:
                tmp = self._qc[x]
            except KeyError:
                #tmp = 0.0
                tmp = zero(self)
            #return tmp if tmp.imag else tmp.real
            return tmp
        else:
            return 0.0
    else:
        return self._qc

def d2c(self, x=None, y=None):
    if (x is not None) and (y is not None):
        if x is y:
            tmp = self.d2(x)
        else:
            if isinstance(x, ADF) and isinstance(y, ADF):
                try:
                    tmp = self._cp[(x, y)]
                except KeyError:
                    try:
                        tmp = self._cp[(y, x)]
                    except KeyError:
                        #tmp = 0.0
                        tmp = zero(self)
            else:
                tmp = 0.0

        return tmp
        #return tmp if tmp.imag else tmp.real

    elif ((x is not None) and not (y is not None)) or \
         ((y is not None) and not (x is not None)):
        return 0.0
    else:
        return self._cp

ADF.d = d
ADF.d2 = d2
ADF.d2c = d2c
