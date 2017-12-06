import autograd
import autograd.numpy as np
from autograd.extend import primitive, defvjp
from .parallel_matmul import _par_matmul

@primitive
def batched_dot(a, b):
    if len(a.shape) != 3 or len(b.shape) != 3 or a.shape[0] != b.shape[0]:
        raise ValueError("a,b must be 3-dimensional arrays, with a.shape[0]==b.shape[0] and a.shape[2]==b.shape[1]")
    elif a.shape[0] == 1:
        ## use numpy.dot for blas
        a = np.reshape(a, a.shape[1:])
        b = np.reshape(b, b.shape[1:])
        c = np.dot(a, b)
        return np.reshape(c, [1] + list(c.shape))
    elif a.shape[2] == 1:
        ## the main cost is simply allocating space for the array,
        ## so we are better off doing things in serial
        a = np.reshape(a, a.shape[:-1])
        b = np.reshape(b, (b.shape[0], b.shape[2]))
        if a.shape[-1] > 1 and b.shape[-1] > 1:
            ## batch outer product
            return np.einsum("ij,ik->ijk", a, b)
        else:
            ## broadcasted elementary-wise multiplication
            outshape = (a.shape[0], a.shape[1], b.shape[1])
            a = np.transpose(a)
            b = np.transpose(b)
            if a.shape[0] == 1:
                a = np.reshape(a, [-1])
            if b.shape[0] == 1:
                b = np.reshape(b, [-1])
            return np.transpose(np.reshape(a*b, outshape[::-1]))
    else:
        ## parallel batched matrix multiply
        return _par_matmul(a, b)

defvjp(
    batched_dot,
    lambda ans, a, b: lambda g: batched_dot(g,
                                            np.transpose(b, (0,2,1))),
    lambda ans, a, b: lambda g: batched_dot(np.transpose(a, (0,2,1)),
                                            g))
#batched_dot.defgrad(lambda ans, a, b: lambda g: batched_dot(g, np.transpose(b, (0,2,1))))
#batched_dot.defgrad(lambda ans, a, b: lambda g: batched_dot(np.transpose(a, (0,2,1)), g), argnum=1)
#batched_dot.defvjp(lambda g, ans, vs, gvs, a, b: batched_dot(
#    g, np.transpose(b, (0,2,1))))
#batched_dot.defvjp(
#    lambda g, ans, vs, gvs, a, b: batched_dot(
#        np.transpose(a, (0,2,1)), g),
#    argnum=1)

def einsum2(*args, **kwargs):
    """
    einsum2(subscripts_str, arr0, arr1)
    or,
    einsum2(op0, subscript_list0, arr1, subscript_list1,
            output_subscript_list)

    This function is similar to einsum, except it only operates
    on two input arrays, does not allow diagonal operations
    (repeated subscripts on the same array), and requires the output
    subscripts to always be specified. Also, when specifying
    subscripts via lists, the subscripts can be arbitrary keys
    (unlike numpy.einsum, where they have to be integers).

    Unlike the standard einsum, einsum2 will perform computations
    in parallel. The number of parallel threads is selected automatically,
    but you can also control this with the environment variable
    OMP_NUM_THREADS.

    To perform the parallel computation, einsum2 will either use
    numpy.dot (if possible), otherwise it will use a parallel
    for loop. The advantage of using numpy.dot is that it
    uses BLAS which is much faster than a for loop. However,
    you need to make sure numpy is compiled against a parallel BLAS
    implementation such as MKL or OpenBlas. You won't need to worry
    about this for most packaged, precompiled versions of numpy
    (e.g. Anaconda Python).
    """
    if isinstance(args[0], str):
        subscripts, a, b = args[:3]
        ab_subs, out_subs = subscripts.split("->")
        a_subs, b_subs = ab_subs.split(",")
        return _einsum2(a, list(a_subs), b, list(b_subs), list(out_subs), *args[3:], **kwargs)
    else:
        return _einsum2(*args, **kwargs)

def _einsum2(a, a_sublist, b, b_sublist, out_sublist):
    for subs in a_sublist, b_sublist, out_sublist:
        if len(subs) != len(set(subs)):
            raise NotImplementedError("Repeated subscripts not implemented")

    a, a_sublist = _sum_unique_axes(a, a_sublist, b_sublist, out_sublist)
    b, b_sublist = _sum_unique_axes(b, b_sublist, a_sublist, out_sublist)

    a_subs, b_subs, out_subs = map(set, (a_sublist, b_sublist, out_sublist))
    if out_subs - (a_subs | b_subs):
        raise ValueError("Output subscripts must be contained within input subscripts")

    a_minus_b = list(a_subs - b_subs)
    b_minus_a = list(b_subs - a_subs)
    # _sum_unique_axes should have removed any axes unique to a,b
    assert set(a_minus_b) <= out_subs and set(b_minus_a) <= out_subs

    ab = a_subs & b_subs
    abc = list(ab & out_subs)
    ab_minus_c = list(ab - out_subs)

    shapes = {}
    for arr,sublist in ((a,a_sublist), (b,b_sublist)):
        # arr.shape breaks in autograd if it has no dimension
        if sublist:
            for i,s in zip(arr.shape, sublist):
                if s not in shapes:
                    shapes[s] = i
                elif shapes[s] != i:
                    raise ValueError("a,b shapes don't match")

    a = _reshape(a, a_sublist, abc, a_minus_b, ab_minus_c)
    b = _reshape(b, b_sublist, abc, ab_minus_c, b_minus_a)
    assert len(a.shape) == len(b.shape) == 3

    c = batched_dot(a, b)

    c_sublist = abc + a_minus_b + b_minus_a
    c = np.reshape(c, [shapes[s] for s in c_sublist])

    return _transpose(c, c_sublist, out_sublist)

def einsum1(in_arr, in_sublist, out_sublist):
    in_arr, in_sublist = _sum_unique_axes(in_arr, in_sublist, out_sublist)
    return _transpose(in_arr, in_sublist, out_sublist)

def _reshape(in_arr, in_sublist, *out_sublists):
    assert len(out_sublists) == 3

    old_sublist = in_sublist
    in_sublist = sum(out_sublists, [])
    in_arr = _transpose(in_arr, old_sublist, in_sublist)

    # in_arr.shape breaks in autograd if it has no dimension
    if in_sublist:
        shapes = {s:i for i,s in zip(in_arr.shape, in_sublist)}
    else: shapes = {}
    return np.reshape(in_arr, [np.prod([shapes[s] for s in out_subs], dtype=int)
                               for out_subs in out_sublists])

def _transpose(in_arr, in_sublist, out_sublist):
    if set(in_sublist) != set(out_sublist):
        raise ValueError("Input and output subscripts don't match")
    for sublist in (in_sublist, out_sublist):
        if len(set(sublist)) != len(sublist):
            raise NotImplementedError("Repeated subscripts not implemented")
    in_idxs = {k:v for v,k in enumerate(in_sublist)}
    return np.transpose(in_arr, axes=[in_idxs[s] for s in out_sublist])

def _sum_unique_axes(in_arr, in_sublist, *keep_subs):
    # assume no repeated subscripts
    assert len(in_sublist) == len(set(in_sublist))

    out_sublist = []
    sum_axes = []
    keep_subs = set([s for ks in keep_subs for s in ks])
    for idx, sub in enumerate(in_sublist):
        if sub in keep_subs:
            out_sublist.append(sub)
        else:
            sum_axes.append(idx)
    if sum_axes:
        return np.sum(in_arr, axis=tuple(sum_axes)), out_sublist
    else:
        return in_arr, out_sublist
