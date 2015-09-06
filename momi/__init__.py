"""
momi (MOran Models for Inference) is a python package for computing the site frequency spectrum,
a summary statistic commonly used in population genetics, and using it to infer demographic history.

momi is under development and the documentation is incomplete.
Please refer to examples/tutorial.py for usage & introduction.

## TODO: finish writing this documentation
"""


from parse_ms import simulate_ms, sfs_list_from_ms, to_ms_cmd, run_ms
from demography import Demography
from sum_product import expected_sfs, expected_total_branch_len, expected_sfs_tensor_prod, expected_tmrca, expected_deme_tmrca
from likelihood_surface import unlinked_log_likelihood, unlinked_mle_search, unlinked_mle_approx_cov, unlinked_log_lik_vector
from util import sum_sfs_list
from simulate_inference import simulate_inference
from tensor import sfs_tensor_prod
