import timeit
import numpy as np

import moran_model

if __name__=="__main__":
    for n in (5, 10, 50, 100, 500):
        moran_model.moran_eigensystem(n)
        for t in 0.001, 0.01, 0.1, 1.0, 10., 50.:
            print("n=%d t:%g" % (n, t))
            print("\tEigen: %g" % timeit.timeit("moran_model.moran_action(1.0, v)", 
                setup="import numpy as np; import moran_model; v = np.random.random(%d + 1)" % n,
                number=10))
            print("\tExpm: %g" % timeit.timeit("moran_model._old_moran_action(1.0, v)", 
                setup="import numpy as np; import moran_model; v = np.random.random(%d + 1)" % n,
                number=10))
