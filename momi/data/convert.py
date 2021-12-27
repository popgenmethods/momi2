import logging
import numpy as np
import pandas as pd
import momi
import json
from collections import Counter
from itertools import product

def sfs_from_dadi(infile, outfile=None):
    """Generate a momi formatted :class:`Sfs` object from the dadi
    frequency spectrum format as documented in section 3.1 of the dadi
    manual. One slight modification is that if you include population
    names after the "folding" flag on the info line, then this
    function will honor these pop names.

    :param str infile: dadi-style sfs to be converted
    :param str,None outfile: Optional output file to write the :class:`Sfs` to

    :rtype: :class:`Sfs`
    """
    dat = open(infile).readlines()
    ## Get rid of comment lines
    dat = [x.strip() for x in dat if not x.startswith("#")]
    if not len(dat) == 3:
        raise Exception("Malformed dadi sfs {}.\n  Must have 3 lines, yours has {}".format(infile, len(dat)))

    ## Parse the info line into nsamps per pop (list of ints), folding flag, and pop names list (if they are given)
    info = dat[0].split()
    nsamps = []
    ## Keep carving off values as long as they cast successfully as int
    for i in info:
        try:
            nsamps.append(int(i))
        except:
            break
    nsamps = np.array(nsamps)
    ## Get rid of quoted pop names if they exist
    pops = [x.replace('"', '') for x in info[len(nsamps)+1:]]
    folded = info[len(nsamps)]
    folded = False if "un" in folded else True
    if not len(pops) == len(nsamps):
        print("Number of populations doesn't agree with number of pop names, using generic names.")
        pops = ["pop"+x for x in range(len(nsamps))]
    logging.getLogger(__name__).info("Info nsamps={} folded={} pops={}".format(nsamps, folded, pops))
    ## Get mask
    mask = list(map(int, dat[2].split()))
    logging.getLogger(__name__).debug("bitmask: {}".format(mask))

    ## Get sfs, and reshape based on sample sizes
    sfs = list(map(float, dat[1].split()))
    logging.getLogger(__name__).debug("SFS from file: {}".format(sfs))
    length = np.ma.array(sfs, mask=mask).sum()
    sfs = np.array(sfs).reshape(nsamps)
    logging.getLogger(__name__).info("length {}".format(length))
    logging.getLogger(__name__).debug("Reshaped sfs: {}".format(sfs))

    ## Get counts per sfs bin
    counts = Counter()
    for sfsbin in product(*[range(y) for y in [x for x in nsamps]]):
        ## Ignore monomorphic snps
        ## nsamps minus 1 here because of the off by one diff between number
        ## of bins in the sfs and actual number of samples
        if sfsbin == tuple(nsamps-1) or sfsbin == tuple([0] * len(nsamps)):
            continue
        ## ignore zero bin counts
        if sfs[sfsbin] == 0:
            continue
        logging.getLogger(__name__).debug(sfsbin, sfs[sfsbin]),
        counts.update({sfsbin:sfs[sfsbin]})
    logging.getLogger(__name__).debug("nbins {}".format(len(counts)))

    ## Convert SFS bin style into momi config style
    configs = pd.DataFrame(index=range(len(counts)), columns=pops)
    
    locus_info = []
    for i, c in enumerate(counts):
        ## (n-1) here because nbins in dadi sfs is n+1
        cfg = np.array([[(n-1)-x, x] for x,n in zip(c, nsamps)])
        configs.iloc[i] = [list(map(int, list(x))) for x in cfg]
        locus_info.append([0, i, counts[c]])
    logging.getLogger(__name__).info("n_snps {}".format(np.sum([x[2] for x in locus_info])))

    ## Finally build the momi style sfs dictionary and write it out
    momi_sfs = {"sampled_pops":pops,
        "folded":folded,
        "length":int(length),
        "configs":configs.values.tolist(),
        "(locus,config_id,count)":locus_info}

    if outfile == None:
        outfile = "./.tmp.sfs"
    with open(outfile, 'w') as out:
        out.write("{}".format(json.dumps(momi_sfs)))
    ## make it pretty
    sfs = momi.Sfs.load(outfile)
    ## Fold if unfolded
    if folded: sfs = sfs.fold()
    sfs.dump(outfile)

    return sfs
