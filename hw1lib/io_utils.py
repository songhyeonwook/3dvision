# -*- coding: utf-8 -*-
"""io_utils — 완성본. 결과 파일 규격 (results/) 저장 · 로드.

규격 (TA 재실행 스크립트가 읽는 파일 — 이름과 키를 바꾸지 마세요):
  results/dataset_context.json    공통    데이터 모드/경로 기록 (다른 데이터의 결과 혼합 방지)
  results/capture_conditions.csv  Part 0  EXIF 표
  results/measurements.json       Part 0  실측값 (mm): box_dims, box_distance, board_distance_image, ref_height, target_height, camera_height_image …
  results/corners_set{A,B,C}.npz  Part 0  검출 코너
  results/K_exif.json             Part 1  {"K": 3x3, "image_size": [W,H], "f_candidates": {...}, "source": "exif"}
  results/part1_distance.json     Part 1  {"image": .., "d_est": .., "d_meas": .., "R": 3x3, "t": [..]}
  results/K_transform_check.json  Part 1  (Part 2 뒤) {"1/2 축소": {"K_pred", "K_recal"}, "90° 회전": {...}}
  results/zhang_set{A,B,C}.json   Part 2  {"K": 3x3, "dist": [k1,k2,p1,p2,k3], "rms": .., "per_view_rms": [...], "std": {...}, "K0": 3x3, "n_images": n, "init": ..}
  results/K.json · dist.json      Part 2  최종(세트 C, scratch) — 다른 Part 가 읽음.  K.json: {"K", "image_size", "source", "std", "rms"}, dist.json: {"dist", "model", "std", "source"}
  results/opencv_setC.json        Part 2  cv2.calibrateCameraExtended 결과 (k1k2 모델 + 5계수 모델), image_size [W,H]
  results/part2_ABC_summary.json  Part 2  세트별 f · σ(f) · RMS · k · 빈 구역 · 초기화
  results/dlt.json                Part 3  {"P": 3x4, "K": 3x3, "R": 3x3, "t", "C": [..], "rms": .., "singular_values", "cond_normalized", "cond_unnormalized", "distance_est": .., "distance_measured": .., "n_points", "image_size": [W,H]}
  results/vp.json                 Part 4  {"vps": {"x": [3], "y": [3], "z": [3]} (동차), "pp": [x,y], "f": [3개 쌍], "f_mean": .., "K_alg82": 3x3, "image", "image_size": [W,H]}
  results/height.json             Part 4  {"H_ref": .., "H_est_horizon_from_vps": .., "H_est_horizon_omega": .., "horizon_angle_diff_deg": .., "H_measured": ..}
  results/driving.json            Part 4  {"vp": [x,y], "pitch_deg": .., "yaw_deg": .., "horizon_row": ..}
  results/undistort_stats.json    Part 5  {"time_ours", "time_cv", "mean_abs_diff", "holes_forward", "max_point_diff", "plumb": {edge: [before, after]}, "mvg_fit": {f: [[k1,k2], rms]}}
  results/f_table.csv             Part 6  method, f_x, f_y, sigma_f, c_x, c_y, source
  results/f_table_context.json    Part 6  f 비교표를 맞춘 공통 영상 크기 [W,H]
  results/part6_camera_position.json · part6_f_distortion.json · part6_cross_ratio.json   Part 6 함정 (c)(d)(e)
"""
import os, json, csv
from pathlib import Path
import numpy as np

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')


def load_config(root):
    """hw1_config.json 에서 모든 Part가 공유하는 DATA/RESULTS 절대 경로를 읽음.

    최소 설정: {"mode": "student", "scale": 1.0}. demo 모드는 data_demo/results_demo.
    student/demo 경로 객체를 추가하면 기본 경로를 바꿀 수 있습니다.
    mode는 student 또는 demo. 두 모드의 data/results 경로는 서로 달라야
    합성 데이터 결과가 학생 제출 폴더에 섞이지 않습니다. IO.RESULTS는
    변경하지 않으므로 호출한 노트북에서 명시적으로 대입하세요.
    DEMO_DATA는 현재 mode와 무관하게 합성 데이터 생성 폴더를 가리킵니다.
    """
    root = Path(root).resolve()
    path = root / 'hw1_config.json'
    if not path.is_file():
        raise FileNotFoundError(f'{path} 가 없습니다. 배포본의 hw1_config.json 을 복원하세요')
    with path.open(encoding='utf8') as fp:
        config = json.load(fp)
    if not isinstance(config, dict):
        raise ValueError('hw1_config.json 은 JSON 객체여야 합니다')
    mode = config.get('mode')
    if mode not in ('student', 'demo'):
        raise ValueError('hw1_config.json 의 mode 는 student 또는 demo 이어야 합니다')
    scale = config.get('scale', 1.0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not np.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('hw1_config.json 의 scale 은 0 초과 1 이하의 숫자여야 합니다')
    defaults = {'student': {'data': 'data', 'results': 'results'},
                'demo': {'data': 'data_demo', 'results': 'results_demo'}}
    resolved = {}
    for name in ('student', 'demo'):
        section = config.get(name, defaults[name])
        if not isinstance(section, dict):
            raise ValueError(f'hw1_config.json 에 {name} 경로 설정이 필요합니다')
        for key in ('data', 'results'):
            value = section.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'hw1_config.json 의 {name}.{key} 는 비어 있지 않은 경로여야 합니다')
            resolved[name, key] = str((root / value).resolve())
    paths = list(resolved.values())
    if len(set(paths)) != len(paths):
        raise ValueError('student/demo 의 data/results 네 경로는 모두 달라야 합니다 (결과 혼합 방지)')
    return dict(DATA=resolved[mode, 'data'], RESULTS=resolved[mode, 'results'], mode=mode, SCALE=float(scale),
                DEMO_DATA=resolved['demo', 'data'])


def rescale_K_for_image(K, saved_size, actual_size):
    """전체 영상을 리사이즈했을 때 저장 K를 현재 이미지 격자로 변환.

    크기는 모두 (W, H), 픽셀 중심은 정수 좌표. A @ K에서
    u_new = sx * (u_old + 0.5) - 0.5 (v도 동일)를 적용합니다.
    가로세로 비율이 다른 입력은 거부합니다 (최대 0.5 px 반올림 허용).
    같은 비율의 크롭은 크기만으로 검출할 수 없으므로, 사용자는 동일한
    전체 시야의 리사이즈에만 써야 합니다. 회전/크롭에는 별도 변환이 필요합니다.
    정규화 카메라 좌표의 왜곡 계수는 이 변환으로 바뀌지 않습니다.
    """
    matrix = np.asarray(K, float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('K 는 유한한 3×3 행렬이어야 합니다')
    old = np.asarray(saved_size, float)
    new = np.asarray(actual_size, float)
    for name, size in [('saved_size', old), ('actual_size', new)]:
        if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0) or np.any(size != np.round(size)):
            raise ValueError(f'{name} 는 양의 정수 [W, H] 이어야 합니다')
    # 가로/세로 각각 0.5 px 반올림할 수 있는 하나의 공통 배율이 존재하는지.
    scale_low = np.max((new - 0.5) / old)
    scale_high = np.min((new + 0.5) / old)
    if scale_low > scale_high + 1e-12:
        raise ValueError(f'영상 비율 불일치: {old.astype(int).tolist()} → {new.astype(int).tolist()}. '
                         '같은 시야/방향의 사진인지 확인하세요. 크롭/회전은 자동 보정하지 않습니다')
    sx, sy = new / old
    A = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
    return A @ matrix


def _np(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: _np(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_np(v) for v in o]
    return o


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf8') as fp:
        json.dump(_np(obj), fp, indent=1, ensure_ascii=False)
    return path


def load_json(path):
    with open(path, encoding='utf8') as fp:
        return json.load(fp)


def rpath(name):
    """results/ 안의 경로."""
    return os.path.join(RESULTS, name)


def save_K(name, K, image_size, source, **extra):
    d = dict(K=np.asarray(K, float), image_size=list(image_size), source=source); d.update(extra)
    return save_json(rpath(name), d)


def load_K(name='K.json'):
    d = load_json(rpath(name))
    return np.array(d['K'], float), d


def load_dist(name='dist.json'):
    d = load_json(rpath(name))
    return np.array(d['dist'], float), d


def write_f_table(rows, path=None):
    """rows: list of dict(method, f_x, f_y, sigma_f, c_x, c_y, source)"""
    path = path or rpath('f_table.csv')
    cols = ['method', 'f_x', 'f_y', 'sigma_f', 'c_x', 'c_y', 'source']
    with open(path, 'w', newline='', encoding='utf8') as fp:
        w = csv.DictWriter(fp, fieldnames=cols, extrasaction='ignore'); w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.3f}' if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
    return path


def read_f_table(path=None):
    path = path or rpath('f_table.csv')
    with open(path, encoding='utf8') as fp:
        return list(csv.DictReader(fp))
