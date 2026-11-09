# -*- coding: utf-8 -*-
"""geometry — 완성본. K · 회전 · 동차좌표 · 정규화 변환 · (왜곡 없는) 투영.

표기 (MVG §6.1):
  X_c = R X_w + t,   x ~ K [R | t] X_w,   정규화 좌표 (x_n, y_n) = (X_c/Z_c, Y_c/Z_c)
  K = [[fx, s, cx], [0, fy, cy], [0, 0, 1]]   (픽셀 좌표 원점은 좌상단)
"""
import numpy as np


# ---------------------------------------------------------------- 내부 파라미터
def make_K(fx, fy, cx, cy, skew=0.0):
    """내부 파라미터 행렬 K (3×3)."""
    return np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)


def K_params(K):
    """K → dict(fx, fy, cx, cy, skew)."""
    return dict(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), skew=float(K[0, 1]))


# ---------------------------------------------------------------- 회전
def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def euler_R(roll=0.0, pitch=0.0, yaw=0.0, deg=True):
    """R = Rz(yaw) Ry(pitch) Rx(roll)."""
    if deg:
        roll, pitch, yaw = np.radians([roll, pitch, yaw])
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def rodrigues(rvec):
    """회전 벡터 (3,) → 회전 행렬 (3×3).  (cv2.Rodrigues 와 같은 규약: 축 = rvec/|rvec|, 각 = |rvec|)"""
    rvec = np.asarray(rvec, dtype=float).reshape(3)
    th = np.linalg.norm(rvec)
    if th < 1e-12:
        return np.eye(3)
    k = rvec / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)


def inv_rodrigues(R):
    """회전 행렬 (3×3) → 회전 벡터 (3,). 각도는 [0, π].

    π 근처에서는 반대칭 부분이 0에 가까워지므로 대칭 부분의 고유벡터로
    회전축을 구합니다. 정확히 π인 회전의 축 부호는 둘 다 같은 회전입니다.
    """
    R = np.asarray(R, dtype=float)
    if R.shape != (3, 3) or not np.isfinite(R).all():
        raise ValueError('R 은 유한한 3×3 회전 행렬이어야 합니다')
    cos_th = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    skew = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    sin_th = np.linalg.norm(skew) / 2
    th = np.arctan2(sin_th, cos_th)
    if th < 1e-8:
        return skew / 2
    if np.pi - th < 1e-5:
        _, axes = np.linalg.eigh((R + R.T) / 2)
        axis = axes[:, -1]
        if sin_th > 1e-12:
            if np.dot(axis, skew) < 0:
                axis = -axis
        elif axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        return th * axis
    return th * skew / (2 * sin_th)


def nearest_rotation(Q):
    """임의의 3×3 행렬에 가장 가까운 회전 행렬 (Frobenius) — SVD: Q = UΣVᵀ → R = UVᵀ  (Zhang 2000 Appendix C)."""
    U, _, Vt = np.linalg.svd(Q)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# ---------------------------------------------------------------- 동차좌표
def to_homog(X):
    """(d, N) → (d+1, N)  마지막 행에 1 추가."""
    X = np.asarray(X, dtype=float)
    return np.vstack([X, np.ones((1, X.shape[1]))])


def from_homog(Xh):
    """(d+1, N) → (d, N)  마지막 성분으로 나눔."""
    Xh = np.asarray(Xh, dtype=float)
    return Xh[:-1] / Xh[-1:]


def hnorm(v):
    """동차 벡터를 마지막 성분이 1이 되도록 (마지막 성분이 0이면 그대로)."""
    v = np.asarray(v, dtype=float)
    return v / v[-1] if abs(v[-1]) > 1e-12 else v


def line_through(p, q):
    """두 점(픽셀 (x, y) 또는 동차 3벡터)을 지나는 직선 l = p × q  (MVG §2.2.1)."""
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    if p.size == 2:
        p = np.append(p, 1.0)
    if q.size == 2:
        q = np.append(q, 1.0)
    return np.cross(p, q)


def intersect(l, m):
    """두 직선의 교점 x = l × m (동차, 정규화하지 않음)  (MVG §2.2.1)."""
    return np.cross(np.asarray(l, dtype=float), np.asarray(m, dtype=float))


# ---------------------------------------------------------------- 투영 (왜곡 없음)
def project_pinhole(K, R, t, Xw):
    """Xw (3, N) 월드 점 → (2, N) 픽셀 (왜곡 없음).  카메라 뒤(Z_c ≤ 0)의 점은 NaN."""
    Xw = np.asarray(Xw, dtype=float)
    Xc = R @ Xw + np.asarray(t, dtype=float).reshape(3, 1)
    x = K @ Xc
    uv = x[:2] / x[2]
    uv[:, Xc[2] <= 0] = np.nan
    return uv


def pixel_to_normalized(K, uv):
    """픽셀 (2, N) → 정규화 좌표 (2, N):  x_n = K⁻¹ x."""
    uv = np.asarray(uv, dtype=float)
    xn = np.linalg.inv(K) @ to_homog(uv)
    return xn[:2] / xn[2]


def normalized_to_pixel(K, xn):
    """정규화 좌표 (2, N) → 픽셀 (2, N)."""
    xn = np.asarray(xn, dtype=float)
    uv = K @ to_homog(xn)
    return uv[:2] / uv[2]


# ---------------------------------------------------------------- 정규화 변환 (DLT 전처리)
def normalization_transform(pts):
    """점 집합 (d, N) → 유사 변환 T (d+1 × d+1): 중심을 원점으로, 원점까지 평균 거리를 √d 로.
    (MVG Algorithm 4.2 · §7.1 "Data normalization")
    반환 T 와 정규화된 점 (d, N)."""
    pts = np.asarray(pts, dtype=float)
    d, N = pts.shape
    c = pts.mean(axis=1, keepdims=True)
    dist = np.sqrt(((pts - c) ** 2).sum(axis=0)).mean()
    s = np.sqrt(d) / dist
    T = np.eye(d + 1)
    T[:d, :d] *= s
    T[:d, d] = (-s * c).ravel()
    return T, from_homog(T @ to_homog(pts))


def skew3(a):
    """[a]× (3×3)."""
    a = np.asarray(a, dtype=float).reshape(3)
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
