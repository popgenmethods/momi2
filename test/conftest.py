import pytest
from adarray import get_order, set_order

''' makes sure adarray has order 2 for test, then resets it'''
@pytest.fixture(scope='function')
def ad_order2(request):
    prev_order = get_order()
    def fin():
        set_order(prev_order)
    request.addfinalizer(fin)

    set_order(2)
