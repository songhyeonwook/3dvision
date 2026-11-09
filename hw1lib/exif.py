# -*- coding: utf-8 -*-
"""exif — 완성본. 스마트폰 JPEG 의 EXIF 읽기 · 촬영 조건표.

주의:
  - 이 모듈의 load_image 는 IMREAD_IGNORE_ORIENTATION 으로 '저장된 픽셀 격자' 그대로 읽습니다. 모든 세트는 같은 방향으로 찍고,
    Orientation 태그가 1 이 아닌 사진이 섞이면 경고합니다 (K 는 저장된 픽셀 격자 기준).
  - DigitalZoomRatio ≠ 1, 해상도 · 초점거리가 사진마다 다르면 세트가 깨진 것입니다.
"""
import os, csv
from pathlib import Path
import numpy as np
from PIL import Image

TAGS = {0x010F: 'Make', 0x0110: 'Model', 0x0112: 'Orientation', 0x829A: 'ExposureTime', 0x829D: 'FNumber',
        0x8827: 'ISO', 0x9003: 'DateTimeOriginal', 0x920A: 'FocalLength', 0xA405: 'FocalLengthIn35mmFilm',
        0xA404: 'DigitalZoomRatio', 0xA002: 'ExifImageWidth', 0xA003: 'ExifImageHeight', 0xA434: 'LensModel',
        0xA433: 'LensMake', 0x9206: 'SubjectDistance'}
IMG_EXT = ('.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG', '.heic', '.HEIC')


def _val(v):
    try:
        if isinstance(v, (int, np.integer)):
            return int(v)
        if hasattr(v, 'denominator'):
            return float(v)
        if isinstance(v, tuple) and len(v) == 2:
            return v[0] / v[1] if v[1] else float('nan')
        if isinstance(v, bytes):
            return v.decode('utf8', 'ignore').strip('\x00')
        return v
    except Exception:
        return str(v)


def read_exif(path):
    """한 장의 EXIF 요약 dict (없는 항목은 None). width/height 는 실제 픽셀 크기."""
    out = {k: None for k in TAGS.values()}
    with Image.open(path) as im:
        out['width'], out['height'] = im.size
        try:
            ex = im.getexif()
            ifd = {}
            try:
                ifd = ex.get_ifd(0x8769)
            except Exception:
                pass
            for tag, name in TAGS.items():
                v = ifd.get(tag, ex.get(tag))
                if v is not None:
                    out[name] = _val(v)
        except Exception:
            pass
    out['file'] = os.path.basename(path)
    return out


def list_images(folder):
    # 폴더 이름의 [, ], *, ? 를 glob 패턴으로 해석하지 않음 (한글 경로도 지원).
    directory = Path(folder)
    if not directory.is_dir():
        return []
    return [str(p) for p in sorted(directory.iterdir()) if p.is_file() and not p.name.startswith('.') and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.heic'}]


def capture_table(folders, csv_path=None):
    """여러 폴더의 사진 EXIF 를 표로. folders: {'setA': 'data/setA', ...}. csv_path 를 주면 저장.
    반환: rows (list of dict)."""
    rows = []
    for name, folder in folders.items():
        for f in list_images(folder):
            r = read_exif(f); r['set'] = name; rows.append(r)
    cols = ['set', 'file', 'Make', 'Model', 'LensModel', 'width', 'height', 'Orientation', 'FocalLength', 'FocalLengthIn35mmFilm',
            'DigitalZoomRatio', 'FNumber', 'ExposureTime', 'ISO', 'DateTimeOriginal']
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf8') as fp:
            w = csv.DictWriter(fp, fieldnames=cols, extrasaction='ignore'); w.writeheader()
            for r in rows:
                w.writerow(r)
    return rows


def check_consistency(rows):
    """세트 전체에서 바뀌면 안 되는 항목(기종 · 해상도 · 초점거리 · 디지털 줌 · 방향)을 점검. 경고 문자열 리스트 반환."""
    warns = []
    if not rows:
        return ['사진이 없습니다 — data/ 폴더 규격을 확인하세요']
    for key in ('Model', 'width', 'height', 'FocalLength', 'Orientation'):
        vals = sorted({str(r.get(key)) for r in rows})
        if len(vals) > 1:
            warns.append(f'{key} 가 사진마다 다릅니다: {vals} — 세트 안에서 같아야 합니다 (K 는 해상도 · 초점거리의 함수)')
    dz = {str(r.get('DigitalZoomRatio')) for r in rows} - {'None', '1.0', '1', '0.0', '0'}
    if dz:
        warns.append(f'DigitalZoomRatio ≠ 1 인 사진이 있습니다: {dz} — 디지털 줌은 크롭 + 리사이즈 = K 가 바뀝니다')
    ori = {str(r.get('Orientation')) for r in rows} - {'None', '1'}
    if ori:
        warns.append(f'EXIF Orientation ≠ 1: {ori} — load_image 는 EXIF 회전을 적용하지 않습니다. 모든 사진이 같은 값이면 괜찮지만 K 는 저장 격자 기준입니다')
    return warns


def load_image(path, gray=False, scale=1.0):
    """cv2.imdecode + IMREAD_IGNORE_ORIENTATION (EXIF 회전 미적용) + 선택적 축소. scale < 1 이면 K 도 함께 바뀜 (Part 1.3)."""
    import cv2
    flag = (cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR) | cv2.IMREAD_IGNORE_ORIENTATION
    im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flag)      # 한글 경로에서도 동작 (cv2.imread 는 Windows 에서 실패)
    if im is None:
        raise FileNotFoundError(f'{path} — 읽을 수 없는 형식(HEIC?)이거나 파일이 없음. JPEG 로 저장해 주세요')
    if scale != 1.0:
        im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return im
