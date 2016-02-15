"""
momi (MOran Models for Inference) is a python package for computing the site frequency spectrum,
a summary statistic commonly used in population genetics, and using it to infer demographic history.

Please refer to examples/tutorial.ipynb for usage & introduction.
"""


from .parse_ms import simulate_ms, sfs_list_from_ms, run_ms, to_ms_cmd, seg_sites_from_ms
from .demography import Demography
from .compute_sfs import expected_sfs, expected_total_branch_len, expected_sfs_tensor_prod, expected_tmrca, expected_deme_tmrca
from .likelihood import composite_log_likelihood, composite_mle_search, composite_log_lik_vector, godambe_scaled_inv, observed_fisher_information, observed_score_covariance
from .util import sum_sfs_list, get_sfs_list, write_seg_sites, read_seg_sites
from .tensor import sfs_tensor_prod
from .data_structure import ConfigList, ObservedSfs, ObservedSfsList
