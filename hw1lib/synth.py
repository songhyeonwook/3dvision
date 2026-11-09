# -*- coding: utf-8 -*-
"""synth — 완성본. 참값(K · 왜곡 · 자세)을 알고 있는 합성 데이터 생성기.

여러분이 작성한 함수가 맞는지 확인하는 용도입니다 (Zhang 2000 §5.1 의 합성 실험과 같은 방식).
  make_synthetic_set(protocol)  : 보드 코너 대응점 (세트 A · B · C 촬영 프로토콜을 흉내)
  render_board_image(...)       : 왜곡까지 넣은 체커보드 '이미지' (코너 검출부터 끝까지 전체 파이프라인 시험용)
  make_synthetic_box(...)       : 치수를 아는 상자의 코너 7개 (Part 3 DLT)
  make_manhattan_lines(...)     : 직교 3방향 평행선의 상 (Part 4 소실점)
  render_driving_frame(...)     : 피치 · 높이를 아는 주행 프레임 (Part 4.5)

규약: X_c = R X_w + t,  왜곡은 정규화 좌표에서 Brown–Conrady (k1, k2, p1, p2, k3) — OpenCV 와 동일.
"""
import numpy as np
from . import geometry as G

# 과제용 체커보드: 10 × 7 칸 (25 mm) → 내부 코너 9 × 6
BOARD_NX, BOARD_NY, BOARD_SQUARE_MM = 9, 6, 25.0

# 합성 카메라 참값 (12 MP 스마트폰을 1/3 로 줄인 정도의 1280 × 960)
K_TRUE = G.make_K(1050.0, 1052.0, 646.3, 473.8)
DIST_TRUE = np.array([-0.085, 0.021, 0.0, 0.0, 0.0])            # k1 k2 만 (접선 0) — 셀프체크의 기준 모델
DIST_TRUE_TANGENTIAL = np.array([-0.085, 0.021, 0.0006, -0.0004, 0.0])  # 접선 왜곡이 있는 '실제 렌즈' 흉내 — 모델 불일치 실험용
IMAGE_SIZE = (1280, 960)  # (W, H)


# ---------------------------------------------------------------- 기본 모델 (참값 생성 전용)
def board_object_points(nx=BOARD_NX, ny=BOARD_NY, square=BOARD_SQUARE_MM):
    """내부 코너의 보드 좌표 (N, 3), Z = 0.  순서는 OpenCV findChessboardCorners 와 같이 행(y) 바깥, 열(x) 안쪽."""
    xs, ys = np.meshgrid(np.arange(nx) * square, np.arange(ny) * square)
    return np.column_stack([xs.ravel(), ys.ravel(), np.zeros(nx * ny)]).astype(float)


def _distort(xn, yn, dist):
    k1, k2, p1, p2, k3 = (list(dist) + [0.0] * 5)[:5]
    r2 = xn * xn + yn * yn
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return xd, yd


def _undistort_iter(xd, yd, dist, iters=12):
    k1, k2, p1, p2, k3 = (list(dist) + [0.0] * 5)[:5]
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (xd - dx) / rad
        y = (yd - dy) / rad
    return x, y


def project_points_true(K, dist, R, t, Xw):
    """참값 투영 (왜곡 포함). Xw (N, 3) → (N, 2).  ※ 여러분의 Part 2 투영 함수와 같은 결과가 나와야 합니다."""
    Xc = (R @ np.asarray(Xw, float).T + np.asarray(t, float).reshape(3, 1))
    xn, yn = Xc[0] / Xc[2], Xc[1] / Xc[2]
    xd, yd = _distort(xn, yn, dist)
    u = K[0, 0] * xd + K[0, 1] * yd + K[0, 2]
    v = K[1, 1] * yd + K[1, 2]
    return np.column_stack([u, v])


# ---------------------------------------------------------------- 보드 자세 생성
def _pose_from_center(K, R, center_uv, depth, board_center):
    """보드 중심이 픽셀 center_uv 에 깊이 depth 로 보이도록 t 를 정함."""
    xn = np.linalg.inv(K) @ np.array([center_uv[0], center_uv[1], 1.0])
    Xc_center = depth * xn / xn[2]
    return Xc_center - R @ board_center


def _random_tilt_R(rng, tilt_deg, inplane_deg):
    axis = rng.normal(size=3); axis[2] = 0.0
    axis /= np.linalg.norm(axis)
    R_tilt = G.rodrigues(axis * np.radians(tilt_deg))
    return R_tilt @ G.rot_z(np.radians(inplane_deg))


def board_poses(protocol, n=15, K=K_TRUE, image_size=IMAGE_SIZE, seed=0, nx=BOARD_NX, ny=BOARD_NY, square=BOARD_SQUARE_MM):
    """세트별 촬영 프로토콜을 흉내낸 n 개의 (R, t).
    'A' 정면 평행(기울기 ≤ 2°, 위치 · 거리만 변화)  — Zhang §4 의 퇴화, 재투영 오차의 함정
    'B' 중앙 집중(기울기 다양, 보드 중심은 이미지 중앙 1/3)
    'C' 권장 프로토콜(3×3 전 구역 + 모서리, 기울기 25–45°, 면내 회전, 거리 혼합)"""
    rng = np.random.default_rng(seed)
    W, H = image_size
    obj = board_object_points(nx, ny, square)
    bc = np.array([(nx - 1) * square / 2, (ny - 1) * square / 2, 0.0])
    poses = []
    grid = [(gx, gy) for gy in (0.25, 0.5, 0.75) for gx in (0.25, 0.5, 0.75)]
    corners = [(0.18, 0.2), (0.82, 0.2), (0.18, 0.8), (0.82, 0.8)]
    i = 0
    tries = 0
    while len(poses) < n and tries < 5000:
        tries += 1
        if protocol == 'A':
            cu = (rng.uniform(0.3, 0.7) * W, rng.uniform(0.3, 0.7) * H)
            depth = rng.uniform(380, 650)
            R = _random_tilt_R(rng, abs(rng.normal(0, 1.2)), rng.uniform(-5, 5))
        elif protocol == 'B':
            cu = (rng.uniform(0.4, 0.6) * W, rng.uniform(0.4, 0.6) * H)
            depth = rng.uniform(450, 750)
            R = _random_tilt_R(rng, rng.uniform(5, 45), rng.uniform(-30, 30))
        elif protocol == 'C':
            k = len(poses)
            if k < 9:
                g = grid[k]; cu = (g[0] * W + rng.normal(0, 0.03 * W), g[1] * H + rng.normal(0, 0.03 * H))
            elif k < 13:
                g = corners[k - 9]; cu = (g[0] * W, g[1] * H)
            else:
                cu = (rng.uniform(0.25, 0.75) * W, rng.uniform(0.25, 0.75) * H)
            depth = rng.uniform(330, 700)
            R = _random_tilt_R(rng, rng.uniform(25, 45), rng.uniform(-45, 45))
        else:
            raise ValueError(protocol)
        t = _pose_from_center(K, R, cu, depth, bc)
        uv = G.project_pinhole(K, R, t, obj.T)
        m = 12
        if np.all(np.isfinite(uv)) and uv[0].min() > m and uv[0].max() < W - m and uv[1].min() > m and uv[1].max() < H - m:
            poses.append((R, t))
            i += 1
    if len(poses) < n:
        raise RuntimeError('자세 생성 실패 — 프로토콜 %s' % protocol)
    return poses


def make_synthetic_set(protocol='C', n=15, K=K_TRUE, dist=DIST_TRUE, image_size=IMAGE_SIZE, noise_px=0.3, seed=0,
                       nx=BOARD_NX, ny=BOARD_NY, square=BOARD_SQUARE_MM):
    """세트 생성. 반환 dict:
      objpoints: [(N,3)] × n (보드 좌표, mm)   imgpoints: [(N,2)] × n (잡음 σ = noise_px 가 더해진 픽셀 좌표)
      R_true, t_true: 참 자세 리스트   K_true, dist_true, image_size, protocol, noise_px"""
    rng = np.random.default_rng(seed + 1000)
    obj = board_object_points(nx, ny, square)
    poses = board_poses(protocol, n, K, image_size, seed, nx, ny, square)
    objpoints, imgpoints, Rs, ts = [], [], [], []
    for R, t in poses:
        uv = project_points_true(K, dist, R, t, obj)
        uv = uv + rng.normal(0, noise_px, uv.shape)
        objpoints.append(obj.copy()); imgpoints.append(uv); Rs.append(R); ts.append(t)
    return dict(objpoints=objpoints, imgpoints=imgpoints, R_true=Rs, t_true=ts, K_true=K.copy(),
                dist_true=np.asarray(dist, float).copy(), image_size=tuple(image_size), protocol=protocol, noise_px=noise_px,
                board=dict(nx=nx, ny=ny, square=square))


# ---------------------------------------------------------------- 보드 이미지 렌더
def render_board_image(K, dist, R, t, image_size=IMAGE_SIZE, nx=BOARD_NX, ny=BOARD_NY, square=BOARD_SQUARE_MM,
                       margin_squares=1.6, supersample=2, blur_sigma=0.7, noise=2.0, seed=0):
    """체커보드 이미지 렌더 (역방향 매핑: 픽셀 → 광선 → 보드 평면 Z = 0 → 무늬).  uint8 (H, W).
    왜곡은 참 모델 그대로 들어가므로 findChessboardCorners → 여러분의 캘리브레이션 → K · dist 복원을 끝까지 시험할 수 있습니다."""
    W, H = image_size
    ss = supersample
    u = (np.arange(W * ss) + 0.5) / ss - 0.5
    v = (np.arange(H * ss) + 0.5) / ss - 0.5
    uu, vv = np.meshgrid(u, v)
    Kinv = np.linalg.inv(K)
    xd = Kinv[0, 0] * uu + Kinv[0, 1] * vv + Kinv[0, 2]
    yd = Kinv[1, 1] * vv + Kinv[1, 2]
    xn, yn = _undistort_iter(xd, yd, dist)
    d = np.stack([xn, yn, np.ones_like(xn)], axis=-1)            # 카메라 좌표 광선
    Rt = R.T
    dw = d @ Rt.T                                                 # Rᵀ d
    cw = -(Rt @ np.asarray(t, float).reshape(3))                  # 카메라 중심 (월드)
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = -cw[2] / dw[..., 2]                                 # Z_w = 0 과의 교점
    Xw = cw[0] + lam * dw[..., 0]
    Yw = cw[1] + lam * dw[..., 1]
    ok = (lam > 0) & np.isfinite(lam)
    # 무늬: 내부 코너 (0,0)…((nx-1)s, (ny-1)s). 칸은 [-s, nx s] × [-s, ny s]
    ix = np.floor(Xw / square); iy = np.floor(Yw / square)
    inside = (Xw >= -square) & (Xw < nx * square) & (Yw >= -square) & (Yw < ny * square)
    checker = ((ix + iy) % 2 == 0)
    img = np.full(uu.shape, 0.62)                                 # 배경(책상)
    bm = margin_squares * square
    paper = (Xw >= -bm) & (Xw < (nx - 1) * square + bm) & (Yw >= -bm) & (Yw < (ny - 1) * square + bm) & ok
    img[paper] = 0.93                                             # 흰 종이
    img[inside & ok & checker] = 0.08
    img[inside & ok & ~checker] = 0.93
    img[~ok] = 0.62
    # 비네팅 + 다운샘플 + 블러 + 잡음
    rr = ((uu - W / 2) / W) ** 2 + ((vv - H / 2) / H) ** 2
    img *= (1 - 0.25 * rr)
    img = img.reshape(H, ss, W, ss).mean(axis=(1, 3))
    try:
        import cv2
        if blur_sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    except ImportError:
        pass
    rng = np.random.default_rng(seed)
    img = img * 255 + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 상자 (Part 3)
BOX_CORNER_NAMES = ['000', '100', '110', '010', '001', '101', '111', '011']


def box_corners(dims):
    """치수 (W, D, H) 의 직육면체 코너 8개 (8, 3). 이름 'xyz' 비트 = 각 축에서 0 또는 1."""
    W, D, H = dims
    return np.array([[0, 0, 0], [W, 0, 0], [W, D, 0], [0, D, 0], [0, 0, H], [W, 0, H], [W, D, H], [0, D, H]], float)


def make_synthetic_box(K=K_TRUE, dist=DIST_TRUE, dims=(300.0, 200.0, 150.0), distance=900.0, elev_deg=30.0, azim_deg=35.0,
                       noise_px=0.5, seed=0, one_face=False):
    """상자 코너의 합성 관측. 카메라는 상자 중심에서 distance 만큼, 고도각 elev · 방위각 azim 에서 중심을 바라봄.
    반환 dict: Xw (M,3) 월드 좌표(mm), uv (M,2) 관측(잡음), names, K_true, dist_true, R_true, t_true, C_true (카메라 중심, 월드)
    one_face=True 이면 윗면 코너 4개만 (퇴화 체험용)."""
    rng = np.random.default_rng(seed)
    X = box_corners(dims)
    center = X.mean(axis=0)
    el, az = np.radians(elev_deg), np.radians(azim_deg)
    C = center + distance * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])  # 월드(z 위)
    # 카메라 축: z 는 C → center, y 는 '아래' (월드 -z 와 가깝게), x = y × z
    zc = (center - C); zc /= np.linalg.norm(zc)
    up = np.array([0, 0, 1.0])
    xc = np.cross(zc, up); xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.vstack([xc, yc, zc])        # 행 = 카메라 축 (월드 표현) → X_c = R (X_w - C)
    t = -R @ C
    names = list(BOX_CORNER_NAMES)
    if one_face:
        idx = [4, 5, 6, 7]
    else:
        # 보이지 않는 코너: 카메라에서 가장 먼 바닥 코너 (상자 뒤·아래)
        depth = (R @ (X - C).T)[2]
        hidden = int(np.argmax(np.where(X[:, 2] == 0, depth, -np.inf)))
        idx = [i for i in range(8) if i != hidden]
    Xw = X[idx]
    uv = project_points_true(K, dist, R, t, Xw) + rng.normal(0, noise_px, (len(idx), 2))
    return dict(Xw=Xw, uv=uv, names=[names[i] for i in idx], K_true=K.copy(), dist_true=np.asarray(dist, float).copy(),
                R_true=R, t_true=t, C_true=C, dims=tuple(dims))


# ---------------------------------------------------------------- 맨해튼 장면 (Part 4)
def make_manhattan_lines(K=K_TRUE, dist=None, n_per_dir=4, extent=3000.0, noise_px=0.5, seed=0, image_size=IMAGE_SIZE,
                         cam_pos=(0.95, 0.95, 0.55), look_at=(0.3, 0.3, 0.35)):
    """방 모서리(직교 3방향 x · y · z)의 평행선 세그먼트의 상. 왜곡 dist=None 이면 핀홀.
    카메라 위치 · 시선은 extent 배수로 지정 (기본: 세 소실점이 모두 유한한 일반 자세).
    반환 dict: segments = [(axis, p (2,), q (2,))], vps_true (3, 2) 픽셀, K_true, R_true, t_true"""
    rng = np.random.default_rng(seed)
    C = np.asarray(cam_pos, float) * extent
    R = _look_at(C, np.asarray(look_at, float) * extent, np.array([0, 0, 1.0]))
    t = -R @ C
    segs = []
    W, H = image_size
    tries = 0
    count = [0, 0, 0]
    while min(count) < n_per_dir and tries < 3000:
        tries += 1
        axis = tries % 3
        if count[axis] >= n_per_dir:
            continue
        base = rng.uniform(0.02, 0.6, 3) * extent
        base[2] = rng.uniform(0.0, 0.8) * extent
        base[axis] = 0.0
        a = base.copy(); b = base.copy()
        lo, hi = (0.0, 0.8) if axis == 2 else (0.0, 0.6)
        a[axis] = rng.uniform(lo, lo + 0.25 * (hi - lo)) * extent; b[axis] = rng.uniform(hi - 0.35 * (hi - lo), hi) * extent
        P = np.vstack([a, b])
        Xc = (R @ P.T + t.reshape(3, 1))
        if np.any(Xc[2] <= 1e-6):
            continue
        uv = G.project_pinhole(K, R, t, P.T).T if dist is None else project_points_true(K, dist, R, t, P)
        if not np.all(np.isfinite(uv)):
            continue
        if uv[:, 0].min() < -0.2 * W or uv[:, 0].max() > 1.2 * W or uv[:, 1].min() < -0.2 * H or uv[:, 1].max() > 1.2 * H:
            continue
        if np.linalg.norm(uv[0] - uv[1]) < 0.03 * W:
            continue
        uv = uv + rng.normal(0, noise_px, uv.shape)
        segs.append((axis, uv[0], uv[1]))
        count[axis] += 1
    vps = []
    for axis in range(3):
        e = np.zeros(3); e[axis] = 1.0
        v = K @ R @ e
        with np.errstate(divide='ignore', invalid='ignore'):
            vps.append(v[:2] / v[2])      # 정면 촬영이면 무한대(inf) 가 나올 수 있음
    return dict(segments=segs, vps_true=np.array(vps), K_true=K.copy(), R_true=R, t_true=t, image_size=image_size)


def _look_at(C, target, up):
    zc = target - C; zc /= np.linalg.norm(zc)
    xc = np.cross(zc, up); xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    return np.vstack([xc, yc, zc])


# ---------------------------------------------------------------- 주행 프레임 (Part 4.5)
def vehicle_to_camera_R():
    """차량 좌표(x 전방, y 좌, z 위) → 카메라 좌표(z 광축, x 우, y 아래)."""
    return np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)


def driving_pose(height=1.45, pitch_deg=-4.0, yaw_deg=1.5):
    """카메라 자세 (R, t): 지면 Z = 0, 카메라 (0, 0, height). 피치 음수 = 아래를 봄."""
    R_look = G.rot_z(np.radians(yaw_deg)) @ G.rot_y(np.radians(-pitch_deg))
    R = vehicle_to_camera_R() @ R_look.T
    C = np.array([0.0, 0.0, height])
    return R, -R @ C


def render_driving_frame(K=G.make_K(1000.0, 1000.0, 640.0, 360.0), image_size=(1280, 720), height=1.45, pitch_deg=-4.0,
                         yaw_deg=1.5, dist=None, lane_half=1.75, seed=0):
    """합성 주행 프레임 (uint8 RGB) 과 참값 dict(pitch_deg, yaw_deg, height, vp_forward (픽셀), horizon_row)."""
    W, H = image_size
    R, t = driving_pose(height, pitch_deg, yaw_deg)
    u, v = np.meshgrid(np.arange(W) + 0.0, np.arange(H) + 0.0)
    Kinv = np.linalg.inv(K)
    xn = Kinv[0, 0] * u + Kinv[0, 2]; yn = Kinv[1, 1] * v + Kinv[1, 2]
    if dist is not None:
        xn, yn = _undistort_iter(xn, yn, dist)
    d = np.stack([xn, yn, np.ones_like(xn)], -1) @ R        # Rᵀ d  (월드 방향)
    C = np.array([0, 0, height])
    with np.errstate(divide='ignore', invalid='ignore'):
        lam = -C[2] / d[..., 2]
    ground = (lam > 0) & np.isfinite(lam)
    Xw = C[0] + lam * d[..., 0]; Yw = C[1] + lam * d[..., 1]
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3))
    sky = np.array([0.72, 0.80, 0.90]); road = np.array([0.33, 0.33, 0.35]); grass = np.array([0.42, 0.50, 0.30])
    img[:] = sky
    road_mask = ground & (np.abs(Yw) < 3 * lane_half + 0.6) & (Xw > 2.0) & (Xw < 160)
    img[ground] = grass
    img[road_mask] = road
    # 차선: 양쪽 실선, 중앙 점선 (3 m 선 · 6 m 간격)
    lw = 0.12
    for y0 in (-lane_half, lane_half):
        m = road_mask & (np.abs(Yw - y0) < lw)
        img[m] = [0.92, 0.92, 0.90]
    dashed = road_mask & (np.abs(Yw) < lw) & ((Xw % 9.0) < 3.0)
    img[dashed] = [0.92, 0.92, 0.90]
    # 바깥 실선
    for y0 in (-3 * lane_half, 3 * lane_half):
        m = road_mask & (np.abs(Yw - y0) < lw)
        img[m] = [0.92, 0.92, 0.90]
    img = np.clip(img * 255 + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    vp = K @ R @ np.array([1.0, 0, 0]); vp = vp[:2] / vp[2]
    truth = dict(pitch_deg=pitch_deg, yaw_deg=yaw_deg, height=height, vp_forward=[float(vp[0]), float(vp[1])],
                 horizon_row=float(vp[1]), K=K.tolist(), image_size=list(image_size))
    return img, truth


# ---------------------------------------------------------------- 방 모서리 장면 렌더 (Part 4 예시 · 데모)
ROOM = dict(size=4500.0, height=2700.0, tile=500.0, door=(1200.0, 2100.0, 2000.0), window=(1500.0, 2700.0, 900.0, 2100.0),
            pole=(1800.0, 1500.0, 1730.0), panel=900.0)


def _hash01(i, j):
    v = np.sin(i * 12.9898 + j * 78.233) * 43758.5453
    return v - np.floor(v)


def room_planes(room=ROOM):
    """바닥(z = 0) · 벽 A(x = 0) · 벽 B(y = 0) — 각 평면: 원점 P0, u 축, v 축, 범위, 텍스처 함수 f(u, v) → 밝기(0–1)."""
    S, H, T = room['size'], room['height'], room['tile']
    dx0, dx1, dh = room['door']; wy0, wy1, wz0, wz1 = room['window']; P = room['panel']

    def floor_tex(u, v):                       # u = x, v = y
        iu, iv = np.floor(u / T), np.floor(v / T)
        g = 0.66 + 0.10 * _hash01(iu, iv)
        grout = (np.abs(u - np.round(u / T) * T) < 7) | (np.abs(v - np.round(v / T) * T) < 7)
        return np.where(grout, 0.30, g)

    def wallA_tex(u, v):                       # x = 0 벽: u = y, v = z
        g = np.full(u.shape, 0.84)
        g = np.where(np.abs(u - np.round(u / P) * P) < 4, 0.55, g)             # 세로 패널 선
        g = np.where((v < 110) & (v > 100), 0.35, g); g = np.where(v <= 100, 0.6, g)   # 걸레받이
        win = (u > wy0) & (u < wy1) & (v > wz0) & (v < wz1)
        g = np.where(win, 0.97, g)
        frame = win & ((np.abs(u - wy0) < 40) | (np.abs(u - wy1) < 40) | (np.abs(v - wz0) < 40) | (np.abs(v - wz1) < 40) | (np.abs(u - (wy0 + wy1) / 2) < 15) | (np.abs(v - (wz0 + wz1) / 2) < 15))
        return np.where(frame, 0.25, g)

    def wallB_tex(u, v):                       # y = 0 벽: u = x, v = z
        g = np.full(u.shape, 0.78)
        g = np.where(np.abs(u - np.round(u / P) * P) < 4, 0.5, g)
        g = np.where(np.abs(v - 2200) < 10, 0.4, g)                                  # 가로 몰딩
        g = np.where((v < 110) & (v > 100), 0.35, g); g = np.where(v <= 100, 0.6, g)
        door = (u > dx0) & (u < dx1) & (v < dh)
        g = np.where(door, 0.48, g)
        frame = door & ((np.abs(u - dx0) < 45) | (np.abs(u - dx1) < 45) | (np.abs(v - dh) < 45))
        g = np.where(frame, 0.22, g)
        knob = door & (np.abs(u - (dx1 - 120)) < 25) & (np.abs(v - 1000) < 25)
        return np.where(knob, 0.1, g)

    return [dict(P0=np.array([0, 0, 0.0]), u=np.array([1, 0, 0.0]), v=np.array([0, 1, 0.0]), ext=(S, S), tex=floor_tex),
            dict(P0=np.array([0, 0, 0.0]), u=np.array([0, 1, 0.0]), v=np.array([0, 0, 1.0]), ext=(S, H), tex=wallA_tex),
            dict(P0=np.array([0, 0, 0.0]), u=np.array([1, 0, 0.0]), v=np.array([0, 0, 1.0]), ext=(S, H), tex=wallB_tex)]


def render_room_scene(K, dist, R, t, image_size=IMAGE_SIZE, room=ROOM, supersample=2, noise=2.5, seed=0):
    """방 모서리(바닥 · 벽 2) 를 역방향 매핑으로 렌더 — 직교 3방향의 직선(타일 줄눈 · 패널 선 · 모서리 · 문 · 창) 이 있는 '건물 사진' 예시.
    반환 uint8 RGB (H, W, 3)."""
    W, H = image_size
    ss = supersample
    u = (np.arange(W * ss) + 0.5) / ss - 0.5; v = (np.arange(H * ss) + 0.5) / ss - 0.5
    uu, vv = np.meshgrid(u, v)
    Kinv = np.linalg.inv(K)
    xd = Kinv[0, 0] * uu + Kinv[0, 1] * vv + Kinv[0, 2]; yd = Kinv[1, 1] * vv + Kinv[1, 2]
    xn, yn = _undistort_iter(xd, yd, dist)
    d = np.stack([xn, yn, np.ones_like(xn)], -1) @ R          # Rᵀ d
    C = -(R.T @ np.asarray(t, float).reshape(3))
    best = np.full(uu.shape, np.inf); img = np.full(uu.shape, 0.55)
    for pl in room_planes(room):
        n = np.cross(pl['u'], pl['v'])
        denom = d @ n
        with np.errstate(divide='ignore', invalid='ignore'):
            lam = ((pl['P0'] - C) @ n) / denom
        X = C + lam[..., None] * d
        pu = (X - pl['P0']) @ pl['u']; pv = (X - pl['P0']) @ pl['v']
        ok = (lam > 1e-6) & (pu >= 0) & (pu <= pl['ext'][0]) & (pv >= 0) & (pv <= pl['ext'][1]) & (lam < best)
        tex = pl['tex'](pu, pv)
        img = np.where(ok, tex, img); best = np.where(ok, lam, best)
    # 거리 감쇠 · 비네팅
    with np.errstate(invalid='ignore'):
        shade = np.where(np.isfinite(best), 1.0 - 0.25 * np.clip(best / 9000.0, 0, 1), 1.0)
    img = img * shade
    rr = ((uu - W / 2) / W) ** 2 + ((vv - H / 2) / H) ** 2
    img *= (1 - 0.18 * rr)
    img = img.reshape(H, ss, W, ss).mean(axis=(1, 3))
    rng = np.random.default_rng(seed)
    g = np.clip(img * 255 + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    rgb = np.stack([g, g, g], -1).astype(np.int16)
    rgb[:, :, 0] = np.clip(rgb[:, :, 0] + 6, 0, 255); rgb[:, :, 2] = np.clip(rgb[:, :, 2] - 8, 0, 255)   # 살짝 따뜻한 톤
    return rgb.astype(np.uint8)


def room_line_segments(room=ROOM):
    """장면 속 직교 3방향 직선(3D 끝점) — 클릭 JSON 생성 · 예시 그림용. {'x': [(a, b), ...], 'y': [...], 'z': [...]}"""
    S, H, T = room['size'], room['height'], room['tile']
    dx0, dx1, dh = room['door']; wy0, wy1, wz0, wz1 = room['window']
    x_lines = [([0, y, 0], [S, y, 0]) for y in (T * 2, T * 4, T * 6)] + [([0, 0, 2200], [S, 0, 2200]), ([0, 0, H], [S, 0, H]), ([dx0, 0, dh], [dx1, 0, dh])]
    y_lines = [([x, 0, 0], [x, S, 0]) for x in (T * 2, T * 4, T * 6)] + [([0, 0, H], [0, S, H]), ([0, wy0, wz0], [0, wy1, wz0]), ([0, wy0, wz1], [0, wy1, wz1])]
    z_lines = [([0, 0, 0], [0, 0, H]), ([dx0, 0, 0], [dx0, 0, dh]), ([dx1, 0, 0], [dx1, 0, dh]), ([0, wy0, wz0], [0, wy0, wz1]), ([0, wy1, wz0], [0, wy1, wz1]), ([room['panel'] * 2, 0, 0], [room['panel'] * 2, 0, H])]
    return {'x': [(np.array(a, float), np.array(b, float)) for a, b in x_lines], 'y': [(np.array(a, float), np.array(b, float)) for a, b in y_lines],
            'z': [(np.array(a, float), np.array(b, float)) for a, b in z_lines]}


def room_cameras(room=ROOM):
    """(R, t) — 'oblique': 세 소실점이 모두 유한한 비스듬한 시점, 'frontal': 벽 A 를 정면으로 보는 시점."""
    S = room['size']
    C1 = np.array([0.80 * S, 0.84 * S, 1700.0]); R1 = _look_at(C1, np.array([0.07 * S, 0.07 * S, 450.0]), np.array([0, 0, 1.0]))
    C2 = np.array([0.85 * S, 0.40 * S, 1300.0]); R2 = _look_at(C2, np.array([0.0, 0.40 * S, 1300.0]), np.array([0, 0, 1.0]))
    return {'oblique': (R1, -R1 @ C1), 'frontal': (R2, -R2 @ C2)}
