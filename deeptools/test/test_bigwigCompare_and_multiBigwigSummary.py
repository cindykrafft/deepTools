import deeptools.bigwigCompare as bwComp
import deeptools.multiBigwigSummary as bwCorr
import numpy as np
import numpy.testing as nt

import os.path
from os import unlink

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/test_data/"
BIGWIG_A = ROOT + "testA_skipNAs.bw"
BIGWIG_B = ROOT + "testB_skipNAs.bw"
BIGWIG_C = ROOT + "test1.bw.bw"


"""
The distribution of reads for the bam file is:

              0                              100                           200
              |------------------------------------------------------------|
testA.bam  3R                                ==============>
                                                            <==============


testB.bam  3R                 <==============               ==============>
                                             ==============>
                                                            ==============>

The resulting bigwig files are as follows:

testA_skipNas:
    3R      100     200     1
    chr_cigar       0       50      2

testB_skipNas:
    3R      50      150     1
    3R      150     200     2
"""


def test_bigwigCompare():
    outfile = '/tmp/result.bg'
    args = "-b1 {} -b2 {} -o {} --operation add --outFileFormat bedgraph".format(BIGWIG_A, BIGWIG_B, outfile).split()
    bwComp.main(args)
    _foo = open(outfile, 'r')
    resp = _foo.readlines()
    _foo.close()
    expected = ['3R\t0\t50\t0\n', '3R\t50\t100\t1\n', '3R\t100\t150\t2\n', '3R\t150\t200\t3\n']
    assert f"{resp}" == f"{expected}", f"{resp} != {expected}"
    unlink(outfile)


def test_bigwigCompare_skipnas():
    outfile = '/tmp/result.bg'
    args = "-b1 {} -b2 {} -o {} --operation add --skipNAs " \
           "--outFileFormat bedgraph".format(BIGWIG_A, BIGWIG_B, outfile).split()
    bwComp.main(args)
    _foo = open(outfile, 'r')
    resp = _foo.readlines()
    _foo.close()
    expected = ['3R\t100\t150\t2\n', '3R\t150\t200\t3\n']
    assert f"{resp}" == f"{expected}", f"{resp} != {expected}"
    unlink(outfile)


def test_bigwigCompare_skipZeroOverZero():
    outfile = '/tmp/result.bg"'
    args = "-b1 {} -b2 {} -o {} --skipZeroOverZero --pseudocount 1 3 --outFileFormat bedgraph".format(BIGWIG_A, BIGWIG_A, outfile).split()
    bwComp.main(args)
    _foo = open(outfile, 'r')
    resp = _foo.readlines()
    _foo.close()
    expected = ['3R\t100\t200\t-1\n']
    assert f"{resp}" == f"{expected}", f"{resp} != {expected}"
    unlink(outfile)


def test_multiBigwigSummary():
    outfile = '/tmp/result.bg'
    args = "bins -b {} {} --binSize 50 -o {}".format(BIGWIG_A, BIGWIG_B, outfile).split()
    bwCorr.main(args)
    resp = np.load(outfile)
    matrix = resp['matrix']
    labels = resp['labels']
    nt.assert_equal(matrix, np.array([[np.nan, np.nan],
                                      [np.nan, 1.],
                                      [1., 1.],
                                      [1., 2.]]))
    nt.assert_equal(labels, ['testA_skipNAs.bw', 'testB_skipNAs.bw'])
    unlink(outfile)


def test_multiBigwigSummary_outrawcounts():
    """
    Test multiBigwigSummary raw counts output
    """
    outfile = '/tmp/result.bg'
    args = "bins -b {} {} --binSize 50 -o /tmp/null --outRawCounts {} ".format(BIGWIG_A, BIGWIG_B, outfile).split()
    bwCorr.main(args)
    _foo = open(outfile, 'r')
    resp = _foo.read()
    _foo.close()
    expected = """#'chr'	'start'	'end'	'testA_skipNAs.bw'	'testB_skipNAs.bw'
3R	0	50	nan	nan
3R	50	100	nan	1.0
3R	100	150	1.0	1.0
3R	150	200	1.0	2.0
"""
    assert f"{resp}" == f"{expected}", f"{resp} != {expected}"
    unlink(outfile)
    unlink("/tmp/null")


def test_multiBigwigSummary_gtf():
    outfile = '/tmp/_test.npz'
    args = "BED-file -b {0} {0} --BED {1}/test.gtf -o {2}".format(BIGWIG_C, ROOT, outfile).split()
    bwCorr.main(args)
    resp = np.load(outfile)
    matrix = resp['matrix']
    labels = resp['labels']
    nt.assert_equal(labels, ['test1.bw.bw', 'test1.bw.bw'])
    nt.assert_allclose(matrix, np.array([[27.475, 27.475],
                                         [27.80669351, 27.80669351]]))
    unlink(outfile)


def test_multiBigwigSummary_metagene():
    outfile = '/tmp/_test.npz'
    args = "BED-file --metagene -b {0} {0} --BED {1}/test.gtf -o {2}".format(BIGWIG_C, ROOT, outfile).split()
    bwCorr.main(args)
    resp = np.load(outfile)
    matrix = resp['matrix']
    labels = resp['labels']
    nt.assert_equal(labels, ['test1.bw.bw', 'test1.bw.bw'])
    nt.assert_allclose(matrix, np.array([[20.28956028, 20.28956028],
                                         [22.1923501, 22.1923501]]))
    unlink(outfile)


def test_multiBigwigSummary_bins_exact_mean():
    """
    Bin means must be the mean of the per-base values, not the zoom-level
    summary (which covers the records overlapping the bin, whole).
    """
    import pyBigWig
    bw = '/tmp/_test_zoom.bw'
    rng = np.random.RandomState(0)
    n = 400000
    vals = rng.gamma(2.0, 1.0, n // 50).repeat(50)          # 50-bp steps, like a bamCoverage track
    vals[100000:102000] += 40.0                              # a peak next to a bin boundary
    fh = pyBigWig.open(bw, 'w')
    fh.addHeader([('chr1', n)])
    fh.addEntries(['chr1'] * (n // 50), list(range(0, n, 50)), ends=list(range(50, n + 1, 50)), values=[float(x) for x in vals[::50]])
    fh.close()
    outfile = '/tmp/_test_zoom.npz'
    args = "bins -b {0} {0} --binSize 10000 -o {1}".format(bw, outfile).split()
    bwCorr.main(args)
    matrix = np.load(outfile)['matrix']
    expected = vals.reshape(-1, 10000).mean(axis=1)
    nt.assert_allclose(np.sort(matrix[:, 0]), np.sort(expected), rtol=1e-6)
    unlink(outfile)
    unlink(bw)
