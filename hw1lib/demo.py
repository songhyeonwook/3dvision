# -*- coding: utf-8 -*-
"""demo — 완성본. 참값을 아는 '합성 데모 데이터셋' 을 data_demo/ 에 학생 폴더 규격 그대로 생성.

    Part 0 노트북의 데모 셀에서 demo.make_demo_dataset('../data_demo')  # 약 3분

본인 사진을 넣기 전에 노트북 전체를 끝까지 돌려 보는 용도입니다 (노트북의 DATA = '../data_demo', RESULTS = '../results_demo').
참값은 data_demo/_truth.json 에 저장됩니다 — 자신의 결과와 비교해 보세요. (실제 제출은 본인 사진 data/ 로.)
"""
import os, json
import numpy as np
from . import synth as S, geometry as G

EXIF_DEMO = dict(Make='Synthetic', Model='HW1 demo cam', FocalLength=4.25, FocalLengthIn35mmFilm=28, DigitalZoomRatio=1.0)


def _save_jpeg(path, img, exif=EXIF_DEMO, quality=95):
    """EXIF 를 넣은 JPEG 저장 (piexif 없으면 EXIF 없이)."""
    from PIL import Image
    im = Image.fromarray(img if img.ndim == 3 else np.repeat(img[:, :, None], 3, axis=2))
    try:
        import piexif
        from fractions import Fraction
        fl = Fraction(exif['FocalLength']).limit_denominator(100)
        zero = {piexif.ImageIFD.Make: exif['Make'], piexif.ImageIFD.Model: exif['Model'], piexif.ImageIFD.Orientation: 1}
        ex = {piexif.ExifIFD.FocalLength: (fl.numerator, fl.denominator), piexif.ExifIFD.FocalLengthIn35mmFilm: int(exif['FocalLengthIn35mmFilm']),
              piexif.ExifIFD.DigitalZoomRatio: (1, 1), piexif.ExifIFD.PixelXDimension: im.size[0], piexif.ExifIFD.PixelYDimension: im.size[1],
              piexif.ExifIFD.FNumber: (18, 10), piexif.ExifIFD.ExposureTime: (1, 120), piexif.ExifIFD.ISOSpeedRatings: 100}
        im.save(path, quality=quality, exif=piexif.dump({'0th': zero, 'Exif': ex}))
    except ImportError:
        im.save(path, quality=quality)


def _draw_polyline(img, pts, color, thickness=2):
    import cv2
    P = np.asarray(pts, float)
    ok = np.all(np.isfinite(P), axis=1)
    P = P[ok]
    for a, b in zip(P[:-1], P[1:]):
        cv2.line(img, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), color, thickness, cv2.LINE_AA)


def _scene_canvas(size, seed=0):
    rng = np.random.default_rng(seed)
    W, H = size
    img = np.full((H, W, 3), 225, np.uint8)
    noise = rng.normal(0, 4, (H, W, 1))
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _shade_box(img, K, dist, R, t, X8):
    """상자의 보이는 면을 면마다 다른 밝기로 채움 (카메라를 향한 면만, 먼 면부터)."""
    import cv2
    faces = [([0, 1, 2, 3], [0, 0, -1.0], 130), ([4, 5, 6, 7], [0, 0, 1.0], 205), ([0, 1, 5, 4], [0, -1, 0.0], 165), ([2, 3, 7, 6], [0, 1, 0.0], 150), ([0, 3, 7, 4], [-1, 0, 0.0], 185), ([1, 2, 6, 5], [1, 0, 0.0], 170)]
    C = -R.T @ np.asarray(t, float).reshape(3)
    vis = []
    for idx, n, g in faces:
        center = X8[idx].mean(axis=0)
        if np.dot(np.asarray(n), C - center) > 0:
            vis.append((np.linalg.norm(C - center), idx, g))
    for _, idx, g in sorted(vis, reverse=True):
        pts = []
        for a, b in zip(idx, idx[1:] + idx[:1]):
            pts.append(_dense_segment(K, dist, R, t, X8[a], X8[b], n=20))
        poly = np.vstack(pts).round().astype(np.int32)
        cv2.fillPoly(img, [poly], (g, g, g), cv2.LINE_AA)


def _dense_segment(K, dist, R, t, a, b, n=60):
    P = np.linspace(0, 1, n)[:, None] * (np.asarray(b, float) - np.asarray(a, float))[None, :] + np.asarray(a, float)[None, :]
    Xc = R @ P.T + t.reshape(3, 1)
    P = P[Xc[2] > 1e-6]
    return S.project_points_true(K, dist, R, t, P)


def make_demo_dataset(root='data_demo', n_per_set=15, seed=0, verbose=True):
    """학생 폴더 규격으로 합성 데이터 생성. 반환: 참값 dict (파일로도 저장)."""
    import cv2
    rng = np.random.default_rng(seed)
    K, dist, size = S.K_TRUE, S.DIST_TRUE, S.IMAGE_SIZE
    truth = dict(K=K.tolist(), dist=dist.tolist(), image_size=list(size), note='합성 데모 참값 — 본인 사진에는 적용되지 않음')
    os.makedirs(root, exist_ok=True)
    # ---- 세트 A · B · C
    poses_all = {}
    for proto in 'ABC':
        folder = os.path.join(root, 'set' + proto); os.makedirs(folder, exist_ok=True)
        poses = S.board_poses(proto, n_per_set, K, size, seed={'A': 1, 'B': 2, 'C': 3}[proto])
        poses_all[proto] = poses
        for i, (R, t) in enumerate(poses):
            img = S.render_board_image(K, dist, R, t, size, seed=100 * ord(proto) + i)
            _save_jpeg(os.path.join(folder, f'{proto}_{i + 1:02d}.jpg'), img)
            if verbose:
                print(f'set{proto} {i + 1}/{n_per_set}', end='\r')
        if verbose:
            print(f'set{proto} 완료      ')
    R0, t0 = poses_all['C'][0]
    C0 = -R0.T @ t0
    truth['setC_01_pose'] = dict(R=R0.tolist(), t=t0.tolist(), C_board=C0.tolist(), dist_to_board_center=float(np.linalg.norm(t0 + R0 @ np.array([100.0, 62.5, 0]))))
    # ---- 상자 (7 코너) · 한 면만 (4 코너)
    bf = os.path.join(root, 'box'); os.makedirs(bf, exist_ok=True)
    box = S.make_synthetic_box(K, dist, dims=(300.0, 200.0, 150.0), distance=900.0, noise_px=0.0, seed=4)
    img = _scene_canvas(size, 5)
    X8 = S.box_corners(box['dims'])
    E = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    _shade_box(img, K, dist, box['R_true'], box['t_true'], X8)
    for a, b in E:
        _draw_polyline(img, _dense_segment(K, dist, box['R_true'], box['t_true'], X8[a], X8[b]), (40, 40, 40), 2)
    _save_jpeg(os.path.join(bf, 'box1.jpg'), img)
    clicks = box['uv'] + rng.normal(0, 0.6, box['uv'].shape)
    json.dump(dict(image='box1.jpg', width=size[0], height=size[1], names=box['names'], points=clicks.tolist()),
              open(os.path.join(bf, 'box1_points.json'), 'w'), indent=1)
    truth['box'] = dict(C_true=box['C_true'].tolist(), R_true=box['R_true'].tolist(), distance_center=float(np.linalg.norm(box['C_true'] - X8.mean(0))))
    box1 = S.make_synthetic_box(K, dist, dims=(300.0, 200.0, 150.0), distance=900.0, elev_deg=75.0, azim_deg=20.0, noise_px=0.0, seed=4, one_face=True)
    img = _scene_canvas(size, 6)
    _shade_box(img, K, dist, box1['R_true'], box1['t_true'], X8)
    for a, b in E:
        _draw_polyline(img, _dense_segment(K, dist, box1['R_true'], box1['t_true'], X8[a], X8[b]), (40, 40, 40), 2)
    _save_jpeg(os.path.join(bf, 'box2_oneface.jpg'), img)
    clicks = box1['uv'] + rng.normal(0, 0.6, box1['uv'].shape)
    json.dump(dict(image='box2_oneface.jpg', width=size[0], height=size[1], names=box1['names'], points=clicks.tolist()),
              open(os.path.join(bf, 'box2_oneface_points.json'), 'w'), indent=1)
    # ---- 건물(방 모서리) 장면: 텍스처 렌더 + 선분 클릭 + plumb-line 점 + 높이(문 = 기준, 막대 = 대상)
    gf = os.path.join(root, 'building'); os.makedirs(gf, exist_ok=True)
    room = S.ROOM
    cams = S.room_cameras(room)
    segs3d = S.room_line_segments(room)
    pole = room['pole']
    poles = dict(ref=([(room['door'][0] + room['door'][1]) / 2, 0.0, 0.0], [(room['door'][0] + room['door'][1]) / 2, 0.0, room['door'][2]]),
                 target=([pole[0], pole[1], 0.0], [pole[0], pole[1], pole[2]]))
    truth['building'] = {}
    for view, fname in (('oblique', 'b1.jpg'), ('frontal', 'b2_frontal.jpg')):
        Rm, tm = cams[view]
        img = S.render_room_scene(K, dist, Rm, tm, size, room, seed=7 if view == 'oblique' else 9).copy()
        # 대상 높이 막대(빨강) — 바닥 위에 선 1730 mm 기둥
        pts = _dense_segment(K, dist, Rm, tm, poles['target'][0], poles['target'][1])
        _draw_polyline(img, pts, (170, 40, 40), 6)
        lines_json = {'x': [], 'y': [], 'z': []}
        plumb = {}
        for axis in 'xyz':
            for a, b in segs3d[axis]:
                p = _dense_segment(K, dist, Rm, tm, a, b)
                if len(p) < 2:
                    continue
                inside = (p[:, 0] > 5) & (p[:, 0] < size[0] - 5) & (p[:, 1] > 5) & (p[:, 1] < size[1] - 5)
                if inside.sum() < 10:
                    continue
                p = p[inside]
                if np.linalg.norm(p[-1] - p[0]) < 0.12 * size[0]:
                    continue
                if len(lines_json[axis]) < 4:
                    lines_json[axis].append([(p[0] + rng.normal(0, 0.7, 2)).tolist(), (p[-1] + rng.normal(0, 0.7, 2)).tolist()])
                if view == 'oblique' and len(plumb) < 3 and np.linalg.norm(p[-1] - p[0]) > 0.45 * size[0]:
                    idx = np.linspace(0, len(p) - 1, 8).astype(int)
                    plumb[f'edge{len(plumb) + 1}'] = (p[idx] + rng.normal(0, 0.5, (8, 2))).tolist()
        _save_jpeg(os.path.join(gf, fname), img)
        json.dump(dict(image=fname, width=size[0], height=size[1], lines=lines_json), open(os.path.join(gf, fname.replace('.jpg', '_lines.json')), 'w'), indent=1)
        vps = []
        for e in np.eye(3):
            v = K @ Rm @ e
            vps.append([float(v[0] / v[2]), float(v[1] / v[2])] if abs(v[2]) > 1e-9 else None)
        truth['building'][view] = dict(image=fname, R_true=Rm.tolist(), t_true=tm.tolist(), vps_true=vps)
        if view == 'oblique':
            json.dump(dict(image=fname, width=size[0], height=size[1], polylines=plumb), open(os.path.join(gf, 'b1_plumb.json'), 'w'), indent=1)
            pts_h = {}
            for nm, (a, b) in poles.items():
                p = _dense_segment(K, dist, Rm, tm, a, b)
                pts_h[nm] = [(p[0] + rng.normal(0, 0.6, 2)).tolist(), (p[-1] + rng.normal(0, 0.6, 2)).tolist()]
            json.dump(dict(image=fname, width=size[0], height=size[1], names=['b_ref', 't_ref', 'b_x', 't_x'],
                           points=[pts_h['ref'][0], pts_h['ref'][1], pts_h['target'][0], pts_h['target'][1]]),
                      open(os.path.join(gf, 'b1_height_points.json'), 'w'), indent=1)
            truth['building'][view].update(H_ref=room['door'][2], H_target=pole[2], ref='door (문 높이)', target='red pole')
    # ---- 주행 프레임
    df = os.path.join(root, 'driving'); os.makedirs(df, exist_ok=True)
    Kd = G.make_K(1000.0, 1000.0, 640.0, 360.0)
    frame, dtruth = S.render_driving_frame(Kd, (1280, 720), height=1.45, pitch_deg=-4.0, yaw_deg=1.5)
    cv2.imwrite(os.path.join(df, 'frame.png'), frame[:, :, ::-1])
    json.dump(dict(K=Kd.tolist(), image_size=[1280, 720], camera_height_m=1.45, note='제공 프레임의 K · 카메라 높이 (피치 · 요는 구할 것)'),
              open(os.path.join(df, 'frame_K.json'), 'w'), indent=1)
    truth['driving'] = dtruth
    # 차선 클릭 흉내: 양쪽 실선 위 두 점씩 (x = 8 m, 40 m)
    Rd, td = S.driving_pose(1.45, -4.0, 1.5)
    lanes = {}
    for key, y0 in (('left', 1.75), ('right', -1.75)):
        P = np.array([[8.0, y0, 0.0], [40.0, y0, 0.0]])
        uv = G.project_pinhole(Kd, Rd, td, P.T).T + rng.normal(0, 0.5, (2, 2))
        lanes[key] = [[uv[0].tolist(), uv[1].tolist()]]      # 선분 1개 (click_tool --lines 규격: 방향 → 선분 목록)
    json.dump(dict(image='frame.png', width=1280, height=720, lines=lanes), open(os.path.join(df, 'frame_lines.json'), 'w'), indent=1)
    # ---- 텍스처 (Part 1.4)
    tex = np.zeros((300, 400, 3), np.uint8)
    gx, gy = np.meshgrid(np.linspace(0, 1, 400), np.linspace(0, 1, 300))
    tex[:, :, 0] = (255 * gx).astype(np.uint8); tex[:, :, 1] = (255 * gy).astype(np.uint8); tex[:, :, 2] = (255 * (1 - gx) * (1 - gy)).astype(np.uint8)
    cv2.putText(tex, '3D VISION', (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.imwrite(os.path.join(root, 'texture.png'), tex[:, :, ::-1])
    # ---- 실측값 (합성이므로 참값 + 작은 '줄자 오차')
    meas = dict(
        note='합성 데모의 실측값 (mm). 본인 데이터는 Part 0 노트북에서 직접 기록',
        monitor_model='(합성)', monitor_native_px=[2560, 1440], monitor_diagonal_inch=27.0, pixel_pitch_mm=0.2335, board_square_px=107, board_square_mm=25.0,
        board_inner_corners=[9, 6],
        box_dims_mm=[300.0, 200.0, 150.0],
        box_distance_mm=float(np.linalg.norm(box['C_true'] - X8.mean(0))) + 3.0, box_distance_note='카메라 렌즈 ~ 상자 중심',
        board_distance_image='C_01.jpg', board_distance_mm=truth['setC_01_pose']['dist_to_board_center'] - 2.0, board_distance_note='카메라 렌즈 ~ 보드 중심',
        camera_height_image='C_01.jpg', camera_height_mm=abs(C0[2]) + 2.5, camera_height_note='보드 평면에서 카메라 렌즈까지의 수직 거리',
        ref_height_mm=2000.0, target_height_mm=1730.0 - 4.0, height_image='b1.jpg', height_note='기준 = 문 높이 2000 mm (b_ref · t_ref = 문 아래 · 위), 대상 = 빨간 막대 (줄자 실측값)')
    json.dump(meas, open(os.path.join(root, 'measurements.json'), 'w'), indent=1, ensure_ascii=False)
    json.dump(truth, open(os.path.join(root, '_truth.json'), 'w'), indent=1, ensure_ascii=False)
    if verbose:
        print('완료:', root)
    return truth


if __name__ == '__main__':
    import sys
    make_demo_dataset(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_demo'))
