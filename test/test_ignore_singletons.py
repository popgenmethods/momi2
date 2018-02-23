import momi
import numpy as np

def test_ignore_singletons():
    demo = momi.demographic_model(1,1)

    demo.add_leaf("A")
    demo.add_leaf("B")
    demo.add_leaf("C")

    demo.add_param("t_ab", .5)
    demo.add_bounded_param("t_bc", lower=["t_ab"], upper=None,
                           x0=.5)
    demo.move_lineages("A", "B", t="t_ab")
    demo.move_lineages("B", "C", t="t_bc")

    data = demo.simulate_data(
        length=1e3, recombination_rate=0,
        mutation_rate=.001, num_replicates=1000,
        sampled_n_dict={"A":1, "B":1, "C":2})

    demo.set_data(data, ignore_singletons=True)

    sfs = demo._get_sfs().exclude_singletons
    esfs = momi.expected_sfs(
        demo._get_demo(), sfs.configs, normalized=True)

    assert np.allclose(esfs.sum(), 1)

    demo.optimize()
    print(demo.get_params())
