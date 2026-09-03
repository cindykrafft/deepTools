import os
import numpy as np
import numpy.testing as nt
import pysam
from deeptools.sumCoveragePerBin import SumCoveragePerBin

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/test_data/"


def test_sum_coverage_partial_bins():
    """
    testB.bam: reads 50-100, 100-150 and twice 150-200 on 3R. With 40-bp bins
    each fragment ends inside a bin; the bin sums are the covered bases
    (0, 30, 20+20, 30+10+10, 40+40), not a whole tile for the last bin.
    """
    c = SumCoveragePerBin([], stepSize=40, binLength=40)
    cov = c.get_coverage_of_region(pysam.AlignmentFile(ROOT + "testB.bam"), '3R', [(0, 200, 40)])
    nt.assert_equal(cov, np.array([0., 30., 40., 50., 80.]))
