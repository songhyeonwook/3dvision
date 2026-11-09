# -*- coding: utf-8 -*-
"""board — 완성본. 체커보드 코너 검출(OpenCV 허용 범위) · 세트 로딩/저장 · 커버리지 맵.

허용 범위 (과제 안내 Part 2.1): cv2.findChessboardCorners(SB) · cornerSubPix · findHomography 는 사용 가능.
그 뒤(B → K → R, t → k → LM)는 직접 구현.
"""
import os
import numpy as np
from . import exif as E
from .synth import board_object_points, BOARD_NX, BOARD_NY, BOARD_SQUARE_MM


def detect_corners(img_gray, pattern=(BOARD_NX, BOARD_NY), subpix=True):
    """(ok, corners (N, 2) float64).  순서: OpenCV 규약(행 바깥 · 열 안쪽). 검출 실패 시 (False, None).
    서브픽셀 창은 코너 간격에 맞춰 자동으로 줄임 — 칸이 작게 찍힌 사진(먼 보드 · 축소 검출)에서 이웃 코너가 창에 들어오면 수 px 가 틀어짐."""
    import cv2
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, c = cv2.findChessboardCorners(img_gray, pattern, flags)
    if not ok:
        try:
            ok, c = cv2.findChessboardCornersSB(img_gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        except Exception:
            ok = False
        if not ok:
            return False, None
        return True, c.reshape(-1, 2).astype(np.float64)        # SB 검출기는 자체 서브픽셀 정밀도 — cornerSubPix 를 덧씌우지 않음
    c = c.reshape(-1, 2)
    if subpix:
        cc = c.reshape(pattern[1], pattern[0], 2)
        spacing = min(np.linalg.norm(np.diff(cc, axis=1), axis=2).min(), np.linalg.norm(np.diff(cc, axis=0), axis=2).min())
        half = int(np.clip(spacing * 0.45, 2, 11))                # 창 반폭: 가장 가까운 이웃 코너의 절반 이하, 최대 11
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
        c = cv2.cornerSubPix(img_gray, np.ascontiguousarray(c, dtype=np.float32), (half, half), (-1, -1), crit)
    return True, c.reshape(-1, 2).astype(np.float64)


def _canonical_order(corners, pattern):
    """검출 순서를 통일: 180° 회전된 순서(첫 코너가 우하단)는 뒤집고, 거울상(왼손) 순서는 열을 뒤집음.
    실제 보드를 앞에서 찍으면 OpenCV 의 순서는 항상 오른손(행 방향 × 열 방향 > 0)이므로 두 번째 규칙은 안전장치."""
    nx, ny = pattern
    c = corners.reshape(ny, nx, 2)
    if c[0, 0, 0] + c[0, 0, 1] > c[-1, -1, 0] + c[-1, -1, 1]:
        c = c[::-1, ::-1]
    d_row = c[0, -1] - c[0, 0]; d_col = c[-1, 0] - c[0, 0]
    if d_row[0] * d_col[1] - d_row[1] * d_col[0] < 0:    # 외적 부호 < 0 = 거울상 순서
        c = c[:, ::-1]
    return c.reshape(-1, 2)


def detect_set(folder, pattern=(BOARD_NX, BOARD_NY), square=BOARD_SQUARE_MM, scale=1.0, verbose=True, max_images=None):
    """폴더의 모든 사진에서 코너 검출. 반환 dict:
      names, imgpoints [(N,2)], objpoints [(N,3) mm], image_size (W,H), failed [names], scale"""
    files = E.list_images(folder)
    if max_images:
        files = files[:max_images]
    obj = board_object_points(pattern[0], pattern[1], square)
    names, imgpts, objpts, failed = [], [], [], []
    size = None
    for f in files:
        g = E.load_image(f, gray=True, scale=scale)
        size = (g.shape[1], g.shape[0])
        ok, c = detect_corners(g, pattern)
        if ok:
            names.append(os.path.basename(f)); imgpts.append(_canonical_order(c, pattern)); objpts.append(obj.copy())
        else:
            failed.append(os.path.basename(f))
        if verbose:
            print(f'{os.path.basename(f):40s} {"OK" if ok else "FAIL"}')
    return dict(names=names, imgpoints=imgpts, objpoints=objpts, image_size=size, failed=failed, scale=scale,
                pattern=pattern, square=square, source_folder=os.path.abspath(folder))


def save_corners(path, det):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez(path, names=np.array(det['names']), imgpoints=np.array(det['imgpoints']), objpoints=np.array(det['objpoints']),
             image_size=np.array(det['image_size']), scale=det['scale'], pattern=np.array(det['pattern']), square=det['square'],
             failed=np.array(det['failed']), source_folder=np.array(det.get('source_folder', '')))


def load_corners(path):
    z = np.load(path, allow_pickle=False)
    return dict(names=list(z['names']), imgpoints=list(z['imgpoints']), objpoints=list(z['objpoints']),
                image_size=tuple(int(v) for v in z['image_size']), scale=float(z['scale']), pattern=tuple(int(v) for v in z['pattern']),
                square=float(z['square']), failed=list(z['failed']),
                source_folder=str(z['source_folder']) if 'source_folder' in z else '')


def coverage_map(imgpoints, image_size, ax=None, title='', grid=(3, 3)):
    """코너 누적 맵 + 3×3 구역별 점 수 (촬영 설계 · 품질 진단용).  반환: 구역별 개수 (3×3 배열)."""
    import matplotlib.pyplot as plt
    W, H = image_size
    allp = np.vstack(imgpoints)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6 * H / W))
    for p in imgpoints:
        ax.plot(p[:, 0], p[:, 1], '.', ms=2, alpha=0.6)
    gx, gy = grid
    counts = np.zeros((gy, gx), int)
    for i in range(gy):
        for j in range(gx):
            m = (allp[:, 0] >= j * W / gx) & (allp[:, 0] < (j + 1) * W / gx) & (allp[:, 1] >= i * H / gy) & (allp[:, 1] < (i + 1) * H / gy)
            counts[i, j] = m.sum()
            ax.text((j + 0.5) * W / gx, (i + 0.5) * H / gy, str(counts[i, j]), ha='center', va='center', fontsize=11, color='k',
                    bbox=dict(fc='w', alpha=0.6, ec='none'))
    for j in range(1, gx):
        ax.axvline(j * W / gx, color='0.6', lw=0.8)
    for i in range(1, gy):
        ax.axhline(i * H / gy, color='0.6', lw=0.8)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect('equal'); ax.set_title(title)
    return counts


def residual_map(imgpoints, residuals, image_size, ax=None, title='', scale=50.0):
    """잔차 벡터장 (관측 − 재투영) — 공간 분포. residuals: [(N,2)] 픽셀."""
    import matplotlib.pyplot as plt
    W, H = image_size
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6 * H / W))
    P = np.vstack(imgpoints); Rr = np.vstack(residuals)
    ax.quiver(P[:, 0], P[:, 1], Rr[:, 0], Rr[:, 1], angles='xy', scale_units='xy', scale=1.0 / scale, width=0.002, color='C3')
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect('equal'); ax.set_title(title + f'  (화살표 {scale:.0f}배)')
    return ax
