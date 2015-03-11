from sum_product import SumProduct
from test_sfs_counts import tree_demo_2, tree_demo_4, admixture_demo

test_demos = [tree_demo_2, tree_demo_4, admixture_demo]

@py.test.mark.parametrize("demo_func", test_demos)
def test_normalizing_constant(demo_func):
    demo, scrm_args, leaf_lins, leaf_pops = demo_func()
