# -*- coding: utf-8 -*-
"""checks — 완성본. 여러분이 작성한 함수를 참값을 아는 합성 데이터로 검증합니다 (정답을 보여주지 않습니다).

각 함수는 (통과 여부, 메시지) 를 돌려주고 한 줄을 출력합니다. 허용 오차는 합성 잡음(σ = 0.3 px) 수준에 맞춘 값입니다.
TA 재실행 스크립트도 같은 검사를 돌립니다 — 통과하지 못한 함수로 만든 결과는 신뢰하지 않습니다.
"""
import numpy as np
from . import synth as S, geometry as G


def _report(name, ok, msg):
    print(('  [OK]   ' if ok else '  [FAIL] ') + name + ' — ' + msg)
    return ok, msg


# ---------------------------------------------------------------- Part 1
def check_K_transform(fn):
    """fn(K, scale, crop_xy, rot90, size) → K'.  축소 · 크롭 · 회전의 참값과 비교 (픽셀 중심 = 정수 좌표 규약)."""
    K = G.make_K(1000.0, 1010.0, 650.0, 470.0)
    W, H = 1280, 960
    X = np.array([[123.0, 456.0, 1.0], [700.0, 200.0, 1.0]]).T  # 픽셀 (동차)
    ok = True; msgs = []
    # 축소 1/2: 픽셀 중심 규약 u' = (u + 0.5)/2 − 0.5
    K2 = fn(K, scale=0.5, crop_xy=(0, 0), rot90=0, size=(W, H))
    exp = np.array([[0.5, 0, 0.5 * 0.5 - 0.5], [0, 0.5, 0.5 * 0.5 - 0.5], [0, 0, 1]]) @ K
    e = np.abs(K2 - exp).max(); ok &= e < 0.3; msgs.append(f'축소 오차 {e:.3f}')
    # 크롭 (100, 50)
    K3 = fn(K, scale=1.0, crop_xy=(100, 50), rot90=0, size=(W - 100, H - 50))
    exp = K.copy(); exp[0, 2] -= 100; exp[1, 2] -= 50
    e = np.abs(K3 - exp).max(); ok &= e < 1e-6; msgs.append(f'크롭 오차 {e:.3f}')
    # 시계방향 90° 회전: (u, v) → (H − 1 − v, u)
    K4 = fn(K, scale=1.0, crop_xy=(0, 0), rot90=1, size=(W, H))
    exp = G.make_K(1010.0, 1000.0, H - 1 - 470.0, 650.0)
    e = np.abs(K4 - exp).max(); ok &= e < 1.01; msgs.append(f'회전 오차 {e:.3f} (1 px 이내 허용: 규약 차이)')
    return _report('K 변환 규칙', ok, ', '.join(msgs))


def check_decompose_H(fn, n=5):
    """fn(K, H) → (R, t).  참 자세로 만든 H = K[r1 r2 t] (스케일 임의) 를 되돌리는지."""
    K = S.K_TRUE
    errs = []
    for R, t in S.board_poses('C', n, K, seed=11):
        H = K @ np.column_stack([R[:, 0], R[:, 1], t]) * 3.7
        R2, t2 = fn(K, H)
        errs.append((np.abs(R2 - R).max(), np.abs(t2 - t).max()))
    e = np.array(errs).max(axis=0)
    ok = e[0] < 1e-6 and e[1] < 1e-4
    return _report('H = K[r1 r2 t] 분해', ok, f'R 오차 {e[0]:.2e}, t 오차 {e[1]:.2e} mm')


# ---------------------------------------------------------------- Part 2
def check_project_points(fn):
    """fn(K, dist, R, t, Xw(N,3)) → (N,2).  참 투영과 비교 (왜곡 포함)."""
    d = S.make_synthetic_set('C', 3, noise_px=0.0, seed=7)
    e = 0.0
    for R, t, X, m in zip(d['R_true'], d['t_true'], d['objpoints'], d['imgpoints']):
        e = max(e, np.abs(fn(d['K_true'], d['dist_true'], R, t, X) - m).max())
    return _report('투영 함수 (왜곡 포함)', e < 1e-6, f'최대 오차 {e:.2e} px')


def check_closed_form(solve_B, K_from_B, noise_px=0.0):
    """solve_B(Hs) → (B, sv), K_from_B(B) → K.  잡음 없는 세트 C 에서 K 복원."""
    import cv2
    d = S.make_synthetic_set('C', 15, dist=np.zeros(5), noise_px=noise_px, seed=3)
    Hs = [cv2.findHomography(X[:, :2], m)[0] for X, m in zip(d['objpoints'], d['imgpoints'])]
    B, _ = solve_B(Hs)
    K = K_from_B(B)
    Kt = d['K_true']
    e = np.abs(K - Kt)
    tol = 1e-3 if noise_px == 0 else 15.0
    ok = e[0, 0] < tol and e[1, 1] < tol and e[0, 2] < tol and e[1, 2] < tol
    return _report('닫힌 해 Vb = 0 → K', ok, f'|Δf| ≤ {max(e[0,0], e[1,1]):.2e}, |Δc| ≤ {max(e[0,2], e[1,2]):.2e} px (잡음 {noise_px} px)')


def check_extrinsics(fn):
    return check_decompose_H(fn)


def check_init_distortion(fn):
    """fn(K, Rs, ts, objpoints, imgpoints) → dist(5,).  참 K · 자세에서 k1 k2 선형 초기화."""
    d = S.make_synthetic_set('C', 15, noise_px=0.0, seed=5)
    k = fn(d['K_true'], d['R_true'], d['t_true'], d['objpoints'], d['imgpoints'])
    e = np.abs(np.ravel(k)[:2] - d['dist_true'][:2])
    ok = e.max() < 0.005
    return _report('왜곡 k1 k2 선형 초기화', ok, f'Δk1 {e[0]:.4f}, Δk2 {e[1]:.4f}')


def check_residuals(fn, pack):
    """fn(p, objpoints, imgpoints) → 잔차 (2N,).  참 파라미터에서 잔차 = 잡음 수준."""
    d = S.make_synthetic_set('C', 5, noise_px=0.0, seed=9)
    p = pack(d['K_true'], d['dist_true'][:2], d['R_true'], d['t_true'])
    r = fn(p, d['objpoints'], d['imgpoints'])
    e = np.abs(r).max()
    return _report('잔차 함수', e < 1e-6 and r.shape == (2 * 5 * 54,), f'참 파라미터에서 최대 잔차 {e:.2e} px, 길이 {r.shape}')


def check_full_pipeline(fn, protocol='C', n=15, noise_px=0.3, seed=1):
    """fn(objpoints, imgpoints, image_size) → dict(K, dist, rms).  잡음 0.3 px 세트에서 f 오차 < 0.5 %, c 오차 < 4 px, k1 오차 < 0.01."""
    d = S.make_synthetic_set(protocol, n, noise_px=noise_px, seed=seed)
    out = fn(d['objpoints'], d['imgpoints'], d['image_size'])
    K, dist = np.asarray(out['K']), np.ravel(out['dist'])
    Kt, dt = d['K_true'], d['dist_true']
    ef = max(abs(K[0, 0] - Kt[0, 0]) / Kt[0, 0], abs(K[1, 1] - Kt[1, 1]) / Kt[1, 1]) * 100
    ec = max(abs(K[0, 2] - Kt[0, 2]), abs(K[1, 2] - Kt[1, 2]))
    ek = abs(dist[0] - dt[0])
    ok = ef < 0.5 and ec < 4.0 and ek < 0.01
    return _report(f'전체 파이프라인 (세트 {protocol})', ok, f'f 오차 {ef:.2f} %, c 오차 {ec:.2f} px, Δk1 {ek:.4f}, RMS {out.get("rms", float("nan")):.3f} px (잡음 {noise_px})')


def check_covariance(fn):
    """fn(J, r) → (Σ, σ).  선형 모델의 알려진 공분산과 비교."""
    rng = np.random.default_rng(0)
    J = rng.normal(size=(400, 3)); r = rng.normal(0, 0.5, 400)
    Sigma, sigma = fn(J, r)
    s2 = (r @ r) / (400 - 3)
    exp = s2 * np.linalg.inv(J.T @ J)
    e = np.abs(Sigma - exp).max()
    return _report('공분산 σ²(JᵀJ)⁻¹', e < 1e-9 and abs(sigma - np.sqrt(s2)) < 1e-9, f'오차 {e:.1e}')


# ---------------------------------------------------------------- Part 3
def check_dlt(dlt_fn, decompose_fn=None):
    """dlt_fn(Xw(N,3), uv(N,2)) → (P, sv).  잡음 없는 상자 7점에서 P 복원 (재투영 < 1e-6)."""
    b = S.make_synthetic_box(dist=np.zeros(5), noise_px=0.0, seed=4)
    P, sv = dlt_fn(b['Xw'], b['uv'])
    x = P @ G.to_homog(b['Xw'].T)
    e = np.linalg.norm((x[:2] / x[2]).T - b['uv'], axis=1).max()
    ok = e < 1e-6
    msg = f'재투영 최대 오차 {e:.2e} px, 최소 특이값 비 {sv[-1]/sv[-2]:.1e}'
    if decompose_fn is not None:
        K, R, t, C = decompose_fn(P)
        eK = np.abs(K / K[2, 2] - b['K_true']).max(); eC = np.linalg.norm(C - b['C_true'])
        ok = ok and eK < 1e-4 and eC < 1e-4
        msg += f', |ΔK| {eK:.1e}, |ΔC| {eC:.1e} mm'
    return _report('정규화 DLT (+ 분해)', ok, msg)


def check_rq(fn):
    """fn(M) → (K, R): K 상삼각 · 대각 양, R 직교 · det +1, K R ∝ M."""
    rng = np.random.default_rng(1)
    ok = True; msgs = []
    for _ in range(5):
        M = rng.normal(size=(3, 3))
        K, R = fn(M)
        KR = K @ R
        s = M[2, 2] / KR[2, 2] if abs(KR[2, 2]) > 1e-12 else 1.0
        e1 = np.abs(KR * s - M).max()
        e2 = np.abs(R @ R.T - np.eye(3)).max()
        tri = abs(K[1, 0]) + abs(K[2, 0]) + abs(K[2, 1])
        pos = np.all(np.diag(K) > 0)
        ok &= e1 < 1e-8 and e2 < 1e-8 and tri < 1e-8 and pos and np.linalg.det(R) > 0
    return _report('RQ 분해 (Givens)', ok, f'재구성 오차 {e1:.1e}, 직교 오차 {e2:.1e}, 상삼각 {tri:.1e}, 대각 양 {pos}')


# ---------------------------------------------------------------- Part 4
def check_vanishing_point(fn):
    """fn(lines (k,3)) → v (3,).  잡음 없는 맨해튼 장면의 소실점."""
    man = S.make_manhattan_lines(noise_px=0.0, seed=3)
    e = 0.0
    for ax in range(3):
        L = [G.line_through(p, q) for a, p, q in man['segments'] if a == ax]
        v = G.hnorm(fn(np.array(L)))[:2]
        e = max(e, np.linalg.norm(v - man['vps_true'][ax]))
    return _report('소실점 최소제곱', e < 1e-3, f'최대 오차 {e:.2e} px')


def check_three_vps(orth_fn, f_fn, K82_fn=None):
    man = S.make_manhattan_lines(noise_px=0.0, seed=3)
    v = man['vps_true']
    pp = orth_fn(v[0], v[1], v[2]); f = f_fn(v[0], v[1], np.asarray(pp))
    Kt = man['K_true']
    e_pp = np.linalg.norm(np.asarray(pp) - Kt[:2, 2]); e_f = abs(f - Kt[0, 0])
    ok = e_pp < 1.5 and e_f < 2.5     # 참 K 는 fx ≠ fy (1050 vs 1052) 이므로 정사각 픽셀 가정에서 ~1 px 차이
    msg = f'주점 오차 {e_pp:.2f} px, f 오차 {e_f:.2f} px'
    if K82_fn is not None:
        vh = np.column_stack([v, np.ones(3)])  # 노트북과 같은 동차 소실점 (3,)
        K82 = K82_fn(vh[0], vh[1], vh[2])
        K82 = K82[0] if isinstance(K82, tuple) else K82
        e82 = max(abs(K82[0, 0] - Kt[0, 0]), abs(K82[0, 2] - Kt[0, 2]), abs(K82[1, 2] - Kt[1, 2]))
        ok = ok and e82 < 2.5; msg += f', Alg 8.2 최대 오차 {e82:.2f} px'
    return _report('세 소실점 → 주점 · f', ok, msg)


def check_transfer_height(fn):
    """fn(b_ref, t_ref, b_x, t_x, v, horizon, H_ref) → H_x."""
    man = S.make_manhattan_lines(noise_px=0.0, seed=3)
    K, R, t = man['K_true'], man['R_true'], man['t_true']
    proj = lambda X: G.project_pinhole(K, R, t, np.array(X, float).reshape(3, 1)).ravel()
    b_ref, t_ref = proj([500, 800, 0]), proj([500, 800, 1000.0])
    b_x, t_x = proj([1500, 400, 0]), proj([1500, 400, 1730.0])
    v = K @ R @ np.array([0, 0, 1.0])
    l = np.linalg.inv(K).T @ (R @ np.array([0, 0, 1.0]))
    Hx = fn(b_ref, t_ref, b_x, t_x, v, l, 1000.0)
    return _report('단일 시점 높이 (Algorithm 8.1)', abs(Hx - 1730.0) < 0.5, f'추정 {Hx:.2f} mm (참 1730)')


def check_pitch(fn):
    img, truth = S.render_driving_frame()
    vp = np.append(np.asarray(truth['vp_forward'], float), 1.0)  # 동차 소실점
    p, y = fn(np.array(truth['K']), vp)
    ok = abs(p - truth['pitch_deg']) < 0.05 and abs(y - truth['yaw_deg']) < 0.05
    return _report('소실점 → 피치 · 요', ok, f'pitch {p:.2f}° (참 {truth["pitch_deg"]}), yaw {y:.2f}° (참 {truth["yaw_deg"]})')


# ---------------------------------------------------------------- Part 5
def check_undistort_backward(fn):
    """fn(img, K, dist) → 보정 영상.  cv2.undistort 와 밝기 차 평균 < 1.0 (보간 차이)."""
    import cv2
    K, d = S.K_TRUE, S.DIST_TRUE
    R, t = S.board_poses('C', 1, seed=5)[0]
    img = S.render_board_image(K, d, R, t)
    ours = np.asarray(fn(img, K, d), float)
    ref = cv2.undistort(img, K, d).astype(float)
    m = ours > 0
    e = np.abs(ours[m] - ref[m]).mean()
    return _report('역방향 언디스토션', e < 1.0 and ours.shape[:2] == img.shape[:2], f'cv2.undistort 대비 평균 밝기 차 {e:.3f}')


def check_undistort_points(fn):
    """fn(uv (N,2), K, dist) → (N,2).  cv2.undistortPoints(P=K) 와 비교."""
    import cv2
    K, d = S.K_TRUE, S.DIST_TRUE
    pts = np.array([[30.0, 40.0], [1250.0, 930.0], [640.0, 480.0], [100.0, 900.0]])
    ours = fn(pts, K, d)
    ref = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, d, P=K).reshape(-1, 2)
    e = np.abs(ours - ref).max()
    return _report('좌표 역왜곡 (반복)', e < 1e-3, f'cv2.undistortPoints 대비 최대 차 {e:.2e} px')


def check_line_residual(fn):
    rng = np.random.default_rng(2)
    t = np.linspace(0, 1, 30)
    pts = np.column_stack([100 + 800 * t, 50 + 300 * t]) + rng.normal(0, 0.5, (30, 2))
    r = fn(pts)
    pts2 = pts.copy(); pts2[:, 1] += 20 * (t - 0.5) ** 2 * 4
    r2 = fn(pts2)
    return _report('직선 곧음 지표', 0.2 < r < 0.9 and r2 > 3 * r, f'직선 {r:.2f} px, 휜 선 {r2:.2f} px')


def run_all(ns):
    """노트북 전역(globals()) 에서 이름으로 함수를 찾아 있는 것만 검사."""
    tests = [('K_after_resize_crop_rotate', lambda: check_K_transform(ns['K_after_resize_crop_rotate'])),
             ('decompose_board_homography', lambda: check_decompose_H(ns['decompose_board_homography'])),
             ('project_points', lambda: check_project_points(ns['project_points'])),
             ('solve_B', lambda: check_closed_form(ns['solve_B'], ns['K_from_B'])),
             ('init_distortion_linear', lambda: check_init_distortion(ns['init_distortion_linear'])),
             ('covariance_from_jacobian', lambda: check_covariance(ns['covariance_from_jacobian'])),
             ('residuals', lambda: check_residuals(ns['residuals'], ns['pack_params']) if 'pack_params' in ns else _report('잔차 함수', False, 'pack_params 가 없음')),
             ('dlt_camera', lambda: check_dlt(ns['dlt_camera'], ns.get('decompose_P'))),
             ('rq_givens', lambda: check_rq(ns['rq_givens'])),
             ('vanishing_point_ls', lambda: check_vanishing_point(ns['vanishing_point_ls'])),
             ('transfer_height', lambda: check_transfer_height(ns['transfer_height'])),
             ('pitch_yaw_from_vp', lambda: check_pitch(ns['pitch_yaw_from_vp'])),
             ('undistort_image_backward', lambda: check_undistort_backward(ns['undistort_image_backward'])),
             ('undistort_pixels', lambda: check_undistort_points(ns['undistort_pixels'])),
             ('line_fit_residual', lambda: check_line_residual(ns['line_fit_residual']))]
    res = {}
    for name, fn in tests:
        if name in ns:
            try:
                res[name] = fn()[0]
            except NotImplementedError:
                print(f'  [TODO] {name} — 아직 구현되지 않음')
            except Exception as e:
                print(f'  [ERR]  {name} — {type(e).__name__}: {e}')
                res[name] = False
    return res
