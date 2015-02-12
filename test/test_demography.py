from demography import Demography
import pytest

@pytest.fixture
def demo():
    test_newick = """
        (a:1[&&momi:model=constant:N=2.0:lineages=10],
         b:1[&&momi:model=constant:N=1.5:lineages=8]):3[&&momi:model=constant:N=10.0];
         """
    return Demography.from_newick(test_newick)


def test_from_newick(demo):
    assert demo.n_lineages_subtended_by[demo.root] == 18

def test_update_state(demo):
    demo.update_state({'a': {'derived': 5, 'ancestral': 5},
                       'b': {'derived': 8, 'ancestral': 0}})
    # Partial updates are allowed
    demo.update_state({'a': {'derived': 5, 'ancestral': 5}})
    with pytest.raises(Exception):
        demo.update_state({'a': {'derived': 5, 'ancestral': 5},
                           'b': {'derived': 4, 'ancestral': 0}})

def test_unsupported_model():
    test_newick = """
        (a:1[&&momi:model=exponential:N=2.0:lineages=10],
         b:1[&&momi:model=constant:N=1.5:lineages=8]):3[&&momi:model=constant:N=10.0];
         """
    with pytest.raises(Exception):
        demo = Demography.from_newick(test_newick)

def test_requires_lineages():
    with pytest.raises(Exception):
        Demography.from_newick("(a:1,b:1)") 
