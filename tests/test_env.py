# -*- coding: utf-8 -*-
"""환경 점검 — `python tests/run_tests.py` 또는 `pytest tests/` 가 통과하면 라이브러리 · OpenCV · 합성 데이터가 정상 (TODO 와 무관)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_imports():
    import cv2, scipy, matplotlib
    from hw1lib import geometry, synth, exif, board, io_utils, checks, click_tool
    assert tuple(int(v) for v in cv2.__version__.split('.')[:2]) >= (4, 5)


def test_synthetic_set_calibrates_with_opencv():
    import cv2
    from hw1lib import synth as S
    d = S.make_synthetic_set('C', 12, noise_px=0.2, seed=3)
    objs = [o.astype(np.float32) for o in d['objpoints']]; imgs = [m.astype(np.float32).reshape(-1, 1, 2) for m in d['imgpoints']]
    rms, K, dist, *_ = cv2.calibrateCamera(objs, imgs, d['image_size'], None, None, flags=cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
    assert abs(K[0, 0] - S.K_TRUE[0, 0]) / S.K_TRUE[0, 0] < 0.01
    assert abs(dist.ravel()[0] - S.DIST_TRUE[0]) < 0.01
    assert rms < 0.5


def test_rendered_board_detects():
    import cv2
    from hw1lib import synth as S, board as Bd
    R, t = S.board_poses('C', 1, seed=5)[0]
    img = S.render_board_image(S.K_TRUE, S.DIST_TRUE, R, t)
    ok, c = Bd.detect_corners(img)
    assert ok and c.shape == (54, 2)
    true_uv = S.project_points_true(S.K_TRUE, S.DIST_TRUE, R, t, S.board_object_points())
    c = Bd._canonical_order(c, (9, 6))
    assert np.abs(c - true_uv).max() < 0.5


def test_geometry_roundtrips():
    from hw1lib import geometry as G
    rng = np.random.default_rng(0)
    for _ in range(5):
        r = rng.normal(size=3)
        assert np.allclose(G.inv_rodrigues(G.rodrigues(r)), r)
    T, p = G.normalization_transform(rng.normal(size=(2, 20)) * 100 + 300)
    assert np.allclose(p.mean(axis=1), 0) and abs(np.linalg.norm(p, axis=0).mean() - np.sqrt(2)) < 1e-9
