# -*- coding: utf-8 -*-
"""click_tool — 완성본. 사진 위에서 점 · 선분을 클릭해 JSON 으로 저장.

노트북 안에서 (기본, ipympl 필요: pip install ipympl):
    get_ipython().run_line_magic('matplotlib', 'widget')
    clicker = CT.NotebookClicker('data/box/box1.jpg', kind='points', names=['000', '100', ...]); clicker.show()
    ...클릭을 마친 뒤 다음 셀에서...  clicker.save('data/box/box1_points.json')
    kind='lines'  (spec={'x': 4, 'y': 4, 'z': 4}: 방향마다 선분 최대 4개, 선분 = 2점. 직선이 모자라면 가운데 클릭 = 그 방향 끝 → 다음 방향)
    kind='polylines' (spec={'edge1': 8, ...}: 에지마다 최대 8점, 가운데 클릭 = 그 에지 끝)
    조작: 왼쪽 클릭 = 점 추가, 오른쪽 클릭 = 마지막 점 취소, 가운데 클릭(휠 클릭) = 현재 방향 · 에지 끝. 툴바 돋보기로 확대한 뒤(돋보기 해제 후) 클릭하면 정밀.

터미널 대안 (ipympl 이 안 될 때):
  python -m hw1lib.click_tool data/box/box1.jpg --points 7 --names 000,100,110,010,001,101,111 --out data/box/box1_points.json
  python -m hw1lib.click_tool data/building/b1.jpg --lines x:4,y:4,z:4 --out data/building/b1_lines.json
  python -m hw1lib.click_tool data/building/b1.jpg --polyline edge1:8,edge2:8,edge3:8 --out data/building/b1_plumb.json

조작: 왼쪽 클릭 = 점 추가, 오른쪽 클릭 = 마지막 점 취소, 가운데 클릭 = 현재 항목 종료(폴리라인), 창 닫기 = 종료.
확대: 툴바의 돋보기로 확대한 뒤 클릭하면 서브픽셀 정밀도로 찍을 수 있습니다 (확대 모드에서는 클릭이 점으로 기록되지 않음 → 돋보기 해제 후 클릭).

JSON 규격:
  points   : {"image": ..., "width": W, "height": H, "names": [...], "points": [[x, y], ...]}
  lines    : {"image": ..., "lines": {"x": [[[x1,y1],[x2,y2]], ...], "y": [...], "z": [...]}}
  polyline : {"image": ..., "polylines": {"edge1": [[x,y], ...], ...}}
좌표는 IMREAD_IGNORE_ORIENTATION 으로 읽은 저장 픽셀 격자 기준 (EXIF 회전 미적용), 픽셀 중심 = 정수 좌표 (OpenCV 규약).
"""
import os, sys, json, argparse
import numpy as np


def _load_rgb(path):
    import cv2
    im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)   # 한글 경로 · 저장 픽셀 격자
    if im is None:
        raise FileNotFoundError(path)
    return im[:, :, ::-1]


def _ginput(ax, fig, n, prompt):
    import matplotlib.pyplot as plt
    pts = []
    ax.set_title(prompt + f'  (0/{n})')
    fig.canvas.draw()
    while len(pts) < n:
        got = plt.ginput(1, timeout=0, mouse_add=1, mouse_pop=3, mouse_stop=2)
        if not got:
            break
        pts.append(got[0])
        ax.plot(got[0][0], got[0][1], 'r+', ms=10, mew=1.2)
        ax.set_title(prompt + f'  ({len(pts)}/{n})')
        fig.canvas.draw()
    return pts


def pick_points(image, n, names=None, title='점 클릭', show=True):
    """노트북/스크립트에서 직접 호출. image: 경로 또는 RGB 배열. 반환 (n, 2) 배열."""
    import matplotlib.pyplot as plt
    rgb = _load_rgb(image) if isinstance(image, str) else image
    fig, ax = plt.subplots(figsize=(11, 11 * rgb.shape[0] / rgb.shape[1]))
    ax.imshow(rgb)
    out = []
    for i in range(n):
        nm = names[i] if names else str(i + 1)
        p = _ginput(ax, fig, 1, f'{title}: {nm}')
        if not p:
            break
        out.append(p[0]); ax.annotate(nm, p[0], color='yellow', fontsize=9)
    plt.close(fig)
    return np.array(out)


def pick_lines(image, spec, title='선분 클릭 (2점)'):
    """spec: {'x': 4, 'y': 4, 'z': 4} → {'x': [[[x1,y1],[x2,y2]], ...], ...}"""
    import matplotlib.pyplot as plt
    rgb = _load_rgb(image) if isinstance(image, str) else image
    fig, ax = plt.subplots(figsize=(11, 11 * rgb.shape[0] / rgb.shape[1]))
    ax.imshow(rgb)
    colors = {'x': 'r', 'y': 'g', 'z': 'b'}
    out = {}
    for key, cnt in spec.items():
        out[key] = []
        for i in range(cnt):
            p = _ginput(ax, fig, 2, f'{title} — 방향 {key} #{i + 1}')
            if len(p) < 2:
                break
            out[key].append([list(p[0]), list(p[1])])
            ax.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], '-', color=colors.get(key, 'y'), lw=1.2)
    plt.close(fig)
    return out


def pick_polylines(image, spec, title='에지 위의 점들 클릭'):
    """spec: {'edge1': 8, ...} → {'edge1': [[x,y], ...]} (가운데 클릭으로 조기 종료)"""
    import matplotlib.pyplot as plt
    rgb = _load_rgb(image) if isinstance(image, str) else image
    fig, ax = plt.subplots(figsize=(11, 11 * rgb.shape[0] / rgb.shape[1]))
    ax.imshow(rgb)
    out = {}
    for key, cnt in spec.items():
        p = _ginput(ax, fig, cnt, f'{title} — {key} (최대 {cnt}점, 가운데 클릭 = 종료)')
        out[key] = [list(q) for q in p]
        if len(p) >= 2:
            P = np.array(p); ax.plot(P[:, 0], P[:, 1], '-o', ms=3, lw=1)
    plt.close(fig)
    return out


class NotebookClicker:
    """노트북 안 클릭 도구 (matplotlib 이벤트, ipympl 백엔드). kind: 'points' | 'lines' | 'polylines'."""

    def __init__(self, image, kind='points', names=None, n=None, spec=None, title=''):
        self.path = image if isinstance(image, str) else None
        self.rgb = _load_rgb(image) if isinstance(image, str) else image
        self.kind = kind
        self.title = title
        if kind == 'points':
            self.names = list(names) if names else [str(i + 1) for i in range(n or 1)]
            self.points = []
        elif kind == 'lines':
            self.spec = dict(spec or {'x': 4, 'y': 4, 'z': 4})
            self.lines = {k: [] for k in self.spec}
            self.cur = []                      # 현재 선분의 첫 점
            self.closed = set()                # 가운데 클릭으로 끝낸 방향 (선분이 spec 보다 적어도 다음 방향으로)
        elif kind == 'polylines':
            self.spec = dict(spec or {'edge1': 8, 'edge2': 8, 'edge3': 8})
            self.polylines = {k: [] for k in self.spec}
            self.closed = set()
        else:
            raise ValueError(kind)
        self.fig = self.ax = None
        self.done = False

    # ---- 진행 상태
    def _status(self):
        if self.kind == 'points':
            i = len(self.points)
            return ('완료' if i >= len(self.names) else f'클릭: {self.names[i]}') + f'  ({i}/{len(self.names)})'
        if self.kind == 'lines':
            for k, n in self.spec.items():
                if k not in self.closed and len(self.lines[k]) < n:
                    return f'방향 {k} 선분 #{len(self.lines[k]) + 1}/{n} — 점 {len(self.cur) + 1}/2 (가운데 클릭 = 이 방향 끝)'
            return '완료'
        for k, n in self.spec.items():
            if k not in self.closed and len(self.polylines[k]) < n:
                return f'{k}: 점 {len(self.polylines[k]) + 1}/{n} (가운데 클릭 = 이 에지 끝)'
        return '완료'

    def _redraw(self):
        ax = self.ax
        for a in list(ax.lines) + list(ax.texts):
            a.remove()
        if self.kind == 'points':
            for i, p in enumerate(self.points):
                ax.plot(p[0], p[1], 'r+', ms=12, mew=1.5); ax.annotate(self.names[i], p, color='yellow', fontsize=9)
        elif self.kind == 'lines':
            cols = {'x': 'r', 'y': 'g', 'z': 'b', 'left': 'y', 'right': 'c'}
            for k, segs in self.lines.items():
                for p, q in segs:
                    ax.plot([p[0], q[0]], [p[1], q[1]], '-', color=cols.get(k, 'm'), lw=1.5)
            for p in self.cur:
                ax.plot(p[0], p[1], 'r+', ms=12, mew=1.5)
        else:
            for k, pts in self.polylines.items():
                if pts:
                    P = np.array(pts); ax.plot(P[:, 0], P[:, 1], '-o', ms=3, lw=1)
        ax.set_title(f'{self.title} {self._status()}', fontsize=10)
        self.fig.canvas.draw_idle()

    def _on_click(self, ev):
        if ev.inaxes != self.ax or ev.xdata is None:
            return
        tb = getattr(self.fig.canvas, 'toolbar', None)
        if tb is not None and getattr(tb, 'mode', '') not in ('', None, 'NONE'):
            return                               # 돋보기 · 이동 모드에서는 클릭을 점으로 받지 않음
        p = [float(ev.xdata), float(ev.ydata)]
        if self.kind == 'points':
            if ev.button == 1 and len(self.points) < len(self.names):
                self.points.append(p)
            elif ev.button == 3 and self.points:
                self.points.pop()
        elif self.kind == 'lines':
            key = next((k for k, n in self.spec.items() if k not in self.closed and len(self.lines[k]) < n), None)
            if ev.button == 1 and key is not None:
                self.cur.append(p)
                if len(self.cur) == 2:
                    self.lines[key].append([self.cur[0], self.cur[1]]); self.cur = []
            elif ev.button == 2 and key is not None:
                if len(self.lines[key]) >= 2:
                    self.closed.add(key); self.cur = []       # 2개 이상이면 이 방향을 끝내고 다음 방향으로
                else:
                    print(f'방향 {key}: 선분이 2개 이상 있어야 끝낼 수 있습니다')
            elif ev.button == 3:
                if self.cur:
                    self.cur.pop()
                else:
                    for k in reversed(list(self.spec)):
                        if self.lines[k]:
                            self.lines[k].pop(); self.closed.discard(k); break
        else:
            key = next((k for k, n in self.spec.items() if k not in self.closed and len(self.polylines[k]) < n), None)
            if ev.button == 3:
                # 에지가 끝나 다음 에지로 넘어갔거나 전체 완료여도 마지막 점을 취소.
                for previous in reversed(list(self.spec)):
                    if self.polylines[previous]:
                        self.polylines[previous].pop()
                        # 취소한 에지부터 다시 입력할 수 있게 종료 상태를 되돌림.
                        keys = list(self.spec)
                        self.closed.difference_update(keys[keys.index(previous):])
                        break
            elif ev.button == 1 and key is not None:
                self.polylines[key].append(p)
            elif ev.button == 2 and key is not None:
                self.closed.add(key)
        self._redraw()

    def show(self, figsize=(10, None)):
        import matplotlib.pyplot as plt
        h, w = self.rgb.shape[:2]
        self.fig, self.ax = plt.subplots(figsize=(figsize[0], figsize[0] * h / w))
        self.ax.imshow(self.rgb); self.ax.set_axis_off()
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self._redraw()
        print('왼쪽 클릭 = 점, 오른쪽 클릭 = 취소' + (', 가운데 클릭 = 에지 끝' if self.kind == 'polylines' else (', 가운데 클릭 = 이 방향 끝(선분 2개 이상일 때)' if self.kind == 'lines' else '')) + ' — 확대는 툴바 돋보기(해제 후 클릭)')
        return self

    def result(self):
        rec = dict(image=os.path.basename(self.path) if self.path else '', width=int(self.rgb.shape[1]), height=int(self.rgb.shape[0]))
        if self.kind == 'points':
            rec.update(names=self.names[:len(self.points)], points=[list(p) for p in self.points])
        elif self.kind == 'lines':
            rec['lines'] = {k: [[list(a), list(b)] for a, b in v] for k, v in self.lines.items()}
        else:
            rec['polylines'] = {k: [list(p) for p in v] for k, v in self.polylines.items()}
        return rec

    def save(self, path):
        rec = self.result()
        if self.kind == 'points' and len(rec['points']) < len(self.names):
            print(f'경고: {len(self.names)}개 중 {len(rec["points"])}개만 클릭됨')
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf8') as fp:
            json.dump(rec, fp, indent=1, ensure_ascii=False)
        print('saved', path)
        return path


def show_points(image, points, names=None, ax=None):
    """저장된 점을 확인용으로 그리기 (노트북 inline)."""
    import matplotlib.pyplot as plt
    rgb = _load_rgb(image) if isinstance(image, str) else image
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 9 * rgb.shape[0] / rgb.shape[1]))
    ax.imshow(rgb)
    P = np.asarray(points, float)
    ax.plot(P[:, 0], P[:, 1], 'r+', ms=10, mew=1.2)
    for i, p in enumerate(P):
        ax.annotate(names[i] if names else str(i + 1), p, color='yellow', fontsize=9)
    ax.set_axis_off()
    return ax


def _select_interactive_backend():
    """사용 가능한 GUI 백엔드를 실제 로딩/figure 생성으로 확인 (CLI 전용)."""
    import importlib
    import matplotlib
    errors = []
    for backend, module in [('QtAgg', 'backend_qtagg'), ('TkAgg', 'backend_tkagg'), ('MacOSX', 'backend_macosx')]:
        try:
            importlib.import_module('matplotlib.backends.' + module)
            matplotlib.use(backend, force=True)
            import matplotlib.pyplot as plt
            figure = plt.figure()
            plt.close(figure)
            return backend
        except Exception as exc:
            errors.append(f'{backend}: {exc}')
    raise RuntimeError('사용 가능한 GUI 백엔드가 없습니다. 노트북의 %matplotlib widget 를 사용하거나 '
                       'Qt/Tk GUI 환경을 설치하세요.\n' + '\n'.join(errors))


def main(argv=None):
    ap = argparse.ArgumentParser(description='HW1 클릭 도구')
    ap.add_argument('image')
    ap.add_argument('--points', type=int, default=0, help='점 개수')
    ap.add_argument('--names', default='', help='점 이름 (쉼표)')
    ap.add_argument('--lines', default='', help='예: x:4,y:4,z:4')
    ap.add_argument('--polyline', default='', help='예: edge1:8,edge2:8')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    _select_interactive_backend()
    rgb = _load_rgb(a.image)
    rec = dict(image=os.path.basename(a.image), width=int(rgb.shape[1]), height=int(rgb.shape[0]))
    if a.points:
        names = a.names.split(',') if a.names else [str(i + 1) for i in range(a.points)]
        P = pick_points(rgb, a.points, names)
        rec.update(names=names[:len(P)], points=P.tolist())
    if a.lines:
        spec = {k: int(v) for k, v in (s.split(':') for s in a.lines.split(','))}
        rec['lines'] = pick_lines(rgb, spec)
    if a.polyline:
        spec = {k: int(v) for k, v in (s.split(':') for s in a.polyline.split(','))}
        rec['polylines'] = pick_polylines(rgb, spec)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w', encoding='utf8') as fp:
        json.dump(rec, fp, indent=1, ensure_ascii=False)
    print('saved', a.out)


if __name__ == '__main__':
    main()
