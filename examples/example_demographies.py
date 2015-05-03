from __future__ import division
from demography import make_demography

## Use autograd.numpy instead of numpy to allow automatic differentiation
import autograd.numpy as anp

def vectorized_demo_func(full_demo_func, n_lins):
    return lambda x: full_demo_func(n_lins, *x)

## TODO: some comments about why we try to make functions be unconstrained

## TODO: add growth, pulse migration
## TODO: rename this function
def simple_human_demo(n_lins_per_pop,
                      t_bottleneck_to_africa_split,
                      t_africa_split_to_eurasia_split,
                      t_eurasia_split_to_present,
                      ancestral_size,
                      africa_size, eurasia_size,
                      eur_present_size, asia_present_size):
    demo_cmd = " ".join(["-I 3 %s" % (" ".join(map(str,n_lins_per_pop))),
                         "-n 1 $0", # present pop size of africa
                         "-n 2 $1", # present pop size of europe
                         "-n 3 $2", # present pop size of asia
                         "-ej $3 3 2 -en $3 2 $4", # eurasia merge and bottleneck
                         "-ej $5 2 1", # eurasia,africa merge
                         "-en $6 1 $7", # ancestral pop size
                         ])
    #ancestral_size=africa_size
    #t_bottleneck_to_africa_split=0.0

    eurasia_split = anp.exp(t_eurasia_split_to_present)
    africa_split = eurasia_split + anp.exp(t_africa_split_to_eurasia_split)
    bottleneck = africa_split + anp.exp(t_bottleneck_to_africa_split)

    demo = make_demography(demo_cmd,
                           anp.exp(africa_size),
                           anp.exp(eur_present_size),
                           anp.exp(asia_present_size),
                           eurasia_split, anp.exp(eurasia_size),
                           africa_split,
                           bottleneck, anp.exp(ancestral_size))
    return demo


## TODO: make a demo that is easier to infer?
## TODO: change ms command line convention
## TODO: does this function even work?
## TODO: function doesn't follow convention of example_human_demo
def example_admixture_demo(x):
    '''
    An example demography with a pulse migration.

    Function is somewhat involved because population labels
    depend on time of split, and because we choose a parametrization
    s.t. the parameter space is all of \mathbb{R}^8

    Input parameter is
    x = np.array([g2,t3,p4,t5,p6,t7,t8,t9])
    where,
    growth rate $2 = g2
    times $3,$5 = exp(t3-1),exp(t5-1)
    probs $4,$6 = 1/(1+p) for p=p4,p6
    times $7,$8 = $3+exp(t7-1), $5+exp(t8-1)
    time $9 = max($7,$8) + exp(t9-1)
    '''
    ms_cmd = ["-I 2 10 10",
              "-g 1 $0", # pop 1 starts with growth rate $2
              "-es $1 1 $2",  # pop 1 pulses to pop $0 at t=$3, w.p. $4
              "-es $3 2 $4", # pop 2 pulses to pop $1 at t=$5, w.p. $6
              "-ej $t5 #3 1", # pops 1 and $0 coalesce at t=$7
              "-ej $t6 #4 2", # pops 2 and $1 coalesce at t=$8
              "-ej $t7 1 2"] # pops 1 and 2 coalesce at t=$9
    ms_cmd = " ".join(ms_cmd)
    
    g0,t1,p2,t3,p4,t5,t6,t7 = x
    
    t1,t3 = map(anp.exp, (t1-1,t3-1))
    p2,p4 = map(lambda p: 1/(1+anp.exp(p)), (p2,p4))
    t5,t6 = t1 + anp.exp(t5-1), t3 + anp.exp(t6-1)
    t7 = anp.maximum(t5,t6) + anp.exp(t7-1)

    demo =  make_demography(ms_cmd,
                            g0,t1,p2,t3,p4,t5=t5,t6=t6,t7=t7)
    return demo
# def example_admixture_demo(x):
#     '''
#     An example demography with a pulse migration.

#     Function is somewhat involved because population labels
#     depend on time of split, and because we choose a parametrization
#     s.t. the parameter space is all of \mathbb{R}^8

#     Input parameter is
#     x = np.array([g2,t3,p4,t5,p6,t7,t8,t9])
#     where,
#     growth rate $2 = g2
#     times $3,$5 = exp(t3-1),exp(t5-1)
#     probs $4,$6 = 1/(1+p) for p=p4,p6
#     times $7,$8 = $3+exp(t7-1), $5+exp(t8-1)
#     time $9 = max($7,$8) + exp(t9-1)
#     '''
#     ms_cmd = ["-I 2 10 10",
#               "-g 1 $2", # pop 1 starts with growth rate $2
#               "-es $3 1 $4",  # pop 1 pulses to pop $0 at t=$3, w.p. $4
#               "-es $5 2 $6", # pop 2 pulses to pop $1 at t=$5, w.p. $6
#               "-ej $7 $0 1", # pops 1 and $0 coalesce at t=$7
#               "-ej $8 $1 2", # pops 2 and $1 coalesce at t=$8
#               "-ej $9 1 2"] # pops 1 and 2 coalesce at t=$9
#     ms_cmd = " ".join(ms_cmd)
    
#     g2,t3,p4,t5,p6,t7,t8,t9 = x
    
#     t3,t5 = map(anp.exp, (t3-1,t5-1))
#     p4,p6 = map(lambda p: 1/(1+anp.exp(p)), (p4,p6))
#     t7,t8 = t3 + anp.exp(t7-1), t5 + anp.exp(t8-1)
#     t9 = anp.maximum(t7,t8) + anp.exp(t9-1)

#     pop0,pop1 = 3,4
#     if t3 > t5:
#         # ms population labels are swapped in this case
#         pop0,pop1 = pop1,pop0

#     demo =  make_demography(ms_cmd,
#                             pop0,pop1,g2,t3,p4,t5,p6,t7,t8,t9)
#     return demo
