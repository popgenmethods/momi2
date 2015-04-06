import ad
from ad import adnumber, ADF, ADV
import numpy as np
from numbers import Number

def d(self, x=None):
    """
    Returns first derivative with respect to x (an AD object).

    Optional
    --------
    x : AD object
        Technically this can be any object, but to make it practically 
        useful, ``x`` should be a single object created using the 
        ``adnumber(...)`` constructor. If ``x=None``, then all associated 
        first derivatives are returned in the form of a ``dict`` object.

    Returns
    -------
    df/dx : scalar
        The derivative (if it exists), otherwise, zero.

    Examples
    --------
    ::
        >>> x = adnumber(2)
        >>> y = 3
        >>> z = x**y

        >>> z.d()
        {ad(2): 12.0}

        >>> z.d(x)
        12.0

        >>> z.d(y)  # derivative wrt y is zero since it's not an AD object
        0.0

    See Also
    --------
    d2, d2c, gradient, hessian

    """
    if x is not None:
        if isinstance(x, ADF):
            try:
                tmp = self._lc[x]
            except KeyError:
                tmp = 0.0
            return tmp
            #return tmp if tmp.imag else tmp.real
        else:
            return 0.0
    else:
        return self._lc

def d2(self, x=None):
        """
        Returns pure second derivative with respect to x (an AD object).
        
        Optional
        --------
        x : AD object
            Technically this can be any object, but to make it practically 
            useful, ``x`` should be a single object created using the 
            ``adnumber(...)`` constructor. If ``x=None``, then all associated 
            second derivatives are returned in the form of a ``dict`` object.
                    
        Returns
        -------
        d2f/dx2 : scalar
            The pure second derivative (if it exists), otherwise, zero.
            
        Examples
        --------
        ::
            >>> x = adnumber(2.5)
            >>> y = 3
            >>> z = x**y
            
            >>> z.d2()
            {ad(2): 15.0}
            
            >>> z.d2(x)
            15.0
            
            >>> z.d2(y)  # second deriv wrt y is zero since not an AD object
            0.0
            
        See Also
        --------
        d, d2c, gradient, hessian
        
        """
        if x is not None:
            if isinstance(x, ADF):
                try:
                    tmp = self._qc[x]
                except KeyError:
                    tmp = 0.0
                #return tmp if tmp.imag else tmp.real
                return tmp
            else:
                return 0.0
        else:
            return self._qc

def d2c(self, x=None, y=None):
        """
        Returns cross-product second derivative with respect to two objects, x
        and y (preferrably AD objects). If both inputs are ``None``, then a dict
        containing all cross-product second derivatives is returned. This is 
        one-way only (i.e., if f = f(x, y) then **either** d2f/dxdy or d2f/dydx
        will be in that dictionary and NOT BOTH). 
        
        If only one of the inputs is ``None`` or if the cross-product 
        derivative doesn't exist, then zero is returned.
        
        If x and y are the same object, then the pure second-order derivative
        is returned.
        
        Optional
        --------
        x : AD object
            Technically this can be any object, but to make it practically 
            useful, ``x`` should be a single object created using the 
            ``adnumber(...)`` constructor.
        y : AD object
            Same as ``x``.
                    
        Returns
        -------
        d2f/dxdy : scalar
            The pure second derivative (if it exists), otherwise, zero.
            
        Examples
        --------
        ::
            >>> x = adnumber(2.5)
            >>> y = adnumber(3)
            >>> z = x**y
            
            >>> z.d2c()
            {(ad(2.5), ad(3)): 33.06704268553368}
            
            >>> z.d2c(x, y)  # either input order gives same result
            33.06704268553368
            
            >>> z.d2c(y, y)  # pure second deriv wrt y
            0.8395887053184748
            
        See Also
        --------
        d, d2, gradient, hessian
        
        """
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
                            tmp = 0.0
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

'''converts a list or np.array of adnumbers into an adfunction of np.arrays'''
class ADArray(ADF):
    def __init__(self, x):
        if isinstance(x, ADF):
            # deep copy
            super(ADArray, self).__init__(x.x, x._lc, x._qc, x._cp)
            return

        if not isinstance(x, np.ndarray):
            x = np.array(x)
        # get the variables to differentiate against
        adentries = []
        for i,xi in np.ndenumerate(x):
            if isinstance(xi, ADF):
                adentries.append((i,xi))
            elif not isinstance(xi,Number):
                raise TypeError(str(xi))
        variables = self._get_variables([xi for _,xi in adentries])

        # initialize the dictionaries of derivatives
        lc_dict, qc_dict, cp_dict = {}, {}, {}
        d_dicts = (lc_dict, qc_dict, cp_dict)
#         for v in variables:
#             for d in d_dicts:
#                 d[v] = np.zeros(x.shape)
        
        # fill the dictionaries of derivatives
        for i,xi in adentries:
            for xi_d, x_d in zip((xi._lc, xi._qc, xi._cp), d_dicts):
                for k in xi_d:
                    if k not in x_d:
                        x_d[k] = np.zeros(x.shape)
                    x_d[k][i] = xi_d[k]

        super(ADArray, self).__init__(x, lc_dict, qc_dict, cp_dict)

    @property
    def shape(self):
        return self.x.shape


def array(x):
    return ADArray(x)

def adapply(f, x):
    ret = array(x)

    ret.x = f(ret.x)
    for d in (x._lc, x._qc, x._cp):
        for v in d:
            d[v] = f(d[v])
    return ret

def adsum(x):
    return adapply(np.sum, x)
