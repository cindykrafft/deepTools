import deeptools.plotCorrelation as pc

import os.path
from os import unlink
from matplotlib.testing.compare import compare_images
import tempfile
import pytest

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/test_data/"
COR_DATA_IN1 = ROOT + "multiBamSummary_result1.npz"
COR_PLOT_1 = ROOT + "plotCorrelation_result1.png"
COR_PLOT_2 = ROOT + "plotCorrelation_result2.png"

COR_PLOT_GG_1 = ROOT + "plotCorrelation_result1_ggplot.png"
COR_PLOT_GG_2 = ROOT + "plotCorrelation_result2_ggplot.png"


@pytest.mark.filterwarnings(
    "ignore:Attempting to set identical low and high xlims:UserWarning"
)
def test_correlation_plot_with_minimal_options():
    """
    Test minimal command line args for correlation plot
    with output as correlation matrix along with matrix
    """
    _, out_matrix = tempfile.mkstemp(suffix=".tsv")
    _, out_png = tempfile.mkstemp(suffix=".png")
    args = "--corData {} -p heatmap -c pearson -o {}  --outFileCorMatrix {}".format(COR_DATA_IN1, out_png, out_matrix).split()
    pc.main(args)

    _foo = open(out_matrix, "r")
    resp = _foo.readlines()[2]
    _foo.close()
    expected = "'bowtie2 test1.bam'\t1.0000\t1.0000\n"
    assert expected in resp, f"'{expected}' not found in '{resp}'"

    res = compare_images(COR_PLOT_1, out_png, 50)
    assert res is None, "Plots do not match"

def test_correlation_plot_scatter():
    """
    Test command line args for correlation plot with output as scatter plot
    """
    _, out_png2 = tempfile.mkstemp(suffix=".png")
    args = "--corData {} -p scatterplot -c pearson  -o {}".format(COR_DATA_IN1, out_png2).split()
    pc.main(args)

    res = compare_images(COR_PLOT_2, out_png2, 50)
    assert res is None, "Plots do not match"

def test_correlation_plot_with_minimal_options_ggplot():
    """
    Test minimal command line args for correlation plot
    with output as correlation matrix along with matrix
    """
    _, out_matrix_gg = tempfile.mkstemp(suffix=".tsv")
    _, out_png_gg = tempfile.mkstemp(suffix=".png")
    args = "--corData {} -p heatmap -c pearson -o {}  --outFileCorMatrix {} --ggplot".format(COR_DATA_IN1, out_png_gg, out_matrix_gg).split()
    pc.main(args)

    _foo = open(out_matrix_gg, "r")
    resp = _foo.readlines()[2]
    _foo.close()
    expected = "'bowtie2 test1.bam'\t1.0000\t1.0000\n"
    assert expected in resp, f"'{expected}' not found in '{resp}'"

    res = compare_images(COR_PLOT_GG_1, out_png_gg, 50)
    assert res is None, "Plots do not match"

def test_correlation_plot_scatter_ggplot():
    """
    Test command line args for correlation plot with output as scatter plot
    """
    _, out_png2_gg = tempfile.mkstemp(suffix=".png")
    args = "--corData {} -p scatterplot -c pearson  -o {} --ggplot".format(COR_DATA_IN1, out_png2_gg).split()
    pc.main(args)

    res = compare_images(COR_PLOT_GG_2, out_png2_gg, 50)
    assert res is None, "Plots do not match"


def test_get_outlier_indices_uses_median_absolute_deviation():
    """--removeOutliers documents the median absolute deviation (MAD) rule with
    a threshold of 200; get_outlier_indices scaled by median(|x|), the median
    itself for counts, so nothing below ~300x the median was ever flagged."""
    import numpy as np
    import numpy.testing as nt
    from deeptools.correlation import Correlation
    rng = np.random.RandomState(1)
    data = rng.poisson(100.0, 5000).astype(float)
    data[[5, 700, 4200]] = 5000.0
    # |x - median| / (1.4826 * MAD) with MAD ~ 7 flags the three planted bins;
    # with median(|x|) in place of the MAD nothing below ~30,000 was flagged
    nt.assert_equal(sorted(Correlation.get_outlier_indices(data)), [5, 700, 4200])
    assert len(Correlation.get_outlier_indices(rng.poisson(100.0, 5000).astype(float))) == 0
