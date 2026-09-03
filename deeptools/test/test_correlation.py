import os
import numpy as np
import deeptools.plotPCA as pca


def _pca_data(extra):
    rng = np.random.RandomState(0)
    mat = rng.poisson(rng.gamma(2.0, 50.0, 300)[:, None] * np.array([1.0, 1.5, 0.5, 2.0]))
    mat[:30, 2:] *= 4
    npz = '/tmp/_test_pca.npz'
    with open(npz, 'wb') as fh:
        np.savez_compressed(fh, matrix=mat, labels=['a', 'b', 'c', 'd'])
    out = '/tmp/_test_pca.tab'
    pca.main('-in {} --outFileNameData {} {}'.format(npz, out, extra).split())
    rows = [line.split('\t') for line in open(out) if not line.startswith('#') and not line.startswith('Component')]
    os.unlink(out)
    os.unlink(npz)
    return mat, np.array([[float(x) for x in r[1:5]] for r in rows])


def _reference(mat, ntop, log2, rowcenter):
    m = mat.astype(float)
    if ntop and m.shape[0] > ntop:
        rvs = m.var(axis=1)
        m = m[np.argpartition(rvs, -ntop)[-ntop:], :]
    if log2:
        m = np.log2(m + 0.01)
    if rowcenter:
        m = m - m.mean(axis=1)[:, None]
    m2 = m - m.mean(axis=0)
    m2 /= np.std(m2, axis=0, ddof=1)
    return np.linalg.svd(m2, full_matrices=False)[2]


def _assert_same_up_to_sign(a, b):
    for i in range(3):
        assert np.allclose(a[i], b[i], atol=1e-6) or np.allclose(a[i], -b[i], atol=1e-6), (a[i], b[i])


def test_plotPCA_log2_and_rowCenter_change_the_result():
    mat, plain = _pca_data('')
    _assert_same_up_to_sign(plain, _reference(mat, 1000, False, False))
    mat, log2 = _pca_data('--log2')
    _assert_same_up_to_sign(log2, _reference(mat, 1000, True, False))
    assert not np.allclose(np.abs(log2[0]), np.abs(plain[0]), atol=1e-6)
    mat, rc = _pca_data('--rowCenter')
    _assert_same_up_to_sign(rc, _reference(mat, 1000, False, True))
    assert not np.allclose(np.abs(rc[0]), np.abs(plain[0]), atol=1e-6)
    mat, top = _pca_data('--ntop 100 --log2 --rowCenter')
    _assert_same_up_to_sign(top, _reference(mat, 100, True, True))
