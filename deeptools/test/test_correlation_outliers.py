import numpy as np
import numpy.testing as nt
from deeptools.correlation import Correlation


def test_get_outlier_indices_uses_median_absolute_deviation():
    rng = np.random.RandomState(1)
    data = rng.poisson(100.0, 5000).astype(float)
    data[[5, 700, 4200]] = 5000.0
    # |x - median| / (1.4826 * MAD) with MAD ~ 7 flags the three planted bins;
    # with median(|x|) in place of the MAD nothing below ~30,000 was flagged
    nt.assert_equal(sorted(Correlation.get_outlier_indices(data)), [5, 700, 4200])
    assert len(Correlation.get_outlier_indices(rng.poisson(100.0, 5000).astype(float))) == 0
