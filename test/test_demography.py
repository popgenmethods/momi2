from demography import Demography

def test_from_newick():
    test_newick = """
        (a:1[&&momi:model=constant:N=2.0:lineages=10],
         b:1[&&momi:model=constant:N=1.5:lineages=8]):3[&&momi:model=constant:N=10.0];
         """
    demo = Demography.from_newick(test_newick)
    assert demo.n_lineages_subtended_by[demo.root] == 18

