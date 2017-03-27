import momi.einsum2 as einsum2
import autograd
import autograd.numpy as np

def test_batched_dot():
    I,J,K,L = np.random.randint(1, 4, size=4)

    A = np.random.normal(size=(I,J,K))
    B = np.random.normal(size=(I,K,L))

    assert np.allclose(einsum2.batched_dot(A,B), A @ B)

def test_einsum2():
    p = .5
    A, Adims = random_tensor(p)
    B, Bdims = random_tensor(p)
    Cdims = np.random.permutation([k for k in set(Adims + Bdims) if np.random.uniform() <= p])

    dim_idxs = {k:i for i,k in enumerate(set(Adims + Bdims))}
    assert np.allclose(einsum2.einsum2(A, Adims, B, Bdims, Cdims),
                       np.einsum(A, [dim_idxs[s] for s in Adims],
                                 B, [dim_idxs[s] for s in Bdims],
                                 [dim_idxs[s] for s in Cdims]))

def test_einsum2_str():
    p = .5
    A, Adims = random_tensor(p)
    B, Bdims = random_tensor(p)
    Cdims = np.random.permutation([k for k in set(Adims + Bdims) if np.random.uniform() <= p])

    dim_idxs = {k:i for i,k in enumerate(set(Adims + Bdims))}
    Adims, Bdims, Cdims = [''.join(dims)
                           for dims in (Adims,Bdims,Cdims)]
    subs = Adims + "," + Bdims + "->" + Cdims
    assert np.allclose(einsum2.einsum2(subs, A, B),
                       einsum2.einsum2(A, Adims,
                                       B, Bdims, Cdims))

def test_grad():
    p = .05
    def fun0(B, Bdims):
        return einsum2.einsum2(np.exp(B**2), Bdims, np.transpose(B), Bdims[::-1], [])
    def fun1(B, Bdims):
        if Bdims: Bdims = list(range(len(Bdims)))
        return np.einsum(np.exp(B**2), Bdims,
                         np.transpose(B), Bdims[::-1], [])
    grad0 = autograd.grad(fun0)
    grad1 = autograd.grad(fun1)
    B, Bdims = random_tensor(p)
    assert np.allclose(grad0(B, Bdims), grad1(B, Bdims))

def random_tensor(p):
    dims = {"a": 1, "b":2, "c":2, "d":3, "e":4}
    Adims = list(np.random.permutation([k for k in dims if np.random.uniform() <= p]))
    A = np.random.normal(size=[dims[s] for s in Adims])
    return A, Adims
