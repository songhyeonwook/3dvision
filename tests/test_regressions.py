# -*- coding: utf-8 -*-
"""배포 라이브러리 회귀 검사. 학생 TODO의 구현은 포함하지 않습니다."""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hw1lib import board as Bd, checks as CK, click_tool as CT, exif as E
from hw1lib import geometry as G, io_utils as IO, synth as S


class GeometryRegressionTests(unittest.TestCase):
    def test_rotation_roundtrip_at_pi_and_near_pi(self):
        from scipy.spatial.transform import Rotation
        axes = np.array([[0, 1, -1], [1, -2, 3], [-3, 0, 1], [0, 0, 1]], float)
        for axis in axes:
            axis /= np.linalg.norm(axis)
            for angle in [0, 1e-10, np.pi - 2e-7, np.pi, np.pi + 2e-7]:
                expected = Rotation.from_rotvec(axis * angle).as_matrix()
                recovered = G.inv_rodrigues(expected)
                with self.subTest(axis=axis, angle=angle):
                    self.assertTrue(np.isfinite(recovered).all())
                    np.testing.assert_allclose(G.rodrigues(recovered), expected, atol=1e-10)

    def test_rescale_K_matches_resize_of_projected_pixels(self):
        K = G.make_K(525, 526, 319.5, 239.5, skew=0.2)
        resized = IO.rescale_K_for_image(K, (640, 480), (1280, 960))
        directions = np.array([[0.1, -0.2, 0.0], [0.3, -0.1, 0.0], [1, 1, 1]], float)
        before = K @ directions
        after = resized @ directions
        np.testing.assert_allclose(after[:2], (before[:2] + 0.5) * 2 - 0.5)
        np.testing.assert_allclose(IO.rescale_K_for_image(resized, (1280, 960), (640, 480)), K)

    def test_rescale_rejects_orientation_crop_and_invalid_sizes(self):
        for old, new in [((640, 480), (480, 640)), ((640, 480), (1280, 720)),
                         ((0, 480), (640, 480)), ((640, 480), (640.5, 480))]:
            with self.subTest(old=old, new=new), self.assertRaises(ValueError):
                IO.rescale_K_for_image(np.eye(3), old, new)
        IO.rescale_K_for_image(np.eye(3), (1001, 751), (500, 376))


class ImageAndConfigRegressionTests(unittest.TestCase):
    def test_list_images_treats_folder_metacharacters_literally(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / '[2026-2] 한글 * 사진?'
            folder.mkdir()
            for name in ['b.JPG', 'a.jpeg', 'note.txt', '.ignored.jpg']:
                (folder / name).write_bytes(b'')
            (folder / 'directory.jpg').mkdir()
            self.assertEqual([Path(p).name for p in E.list_images(folder)], ['a.jpeg', 'b.JPG'])
            self.assertEqual(E.list_images(folder / 'missing'), [])

    def test_exif_orientation_preserves_stored_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            pixels = np.zeros((40, 80, 3), dtype=np.uint8)
            pixels[:, :40, 0] = 255
            for orientation in [1, 3, 6, 8]:
                path = Path(temp) / f'orientation{orientation}.jpg'
                exif = Image.Exif()
                exif[274] = orientation
                Image.fromarray(pixels).save(path, exif=exif)
                with self.subTest(orientation=orientation):
                    self.assertEqual((E.read_exif(path)['width'], E.read_exif(path)['height']), (80, 40))
                    color = E.load_image(str(path))
                    self.assertEqual(color.shape, (40, 80, 3))
                    self.assertEqual(E.load_image(str(path), gray=True).shape, (40, 80))
                    np.testing.assert_array_equal(CT._load_rgb(str(path)), color[:, :, ::-1])
                    self.assertGreater(color[20, 10, 2], color[20, 70, 2])

    def test_config_modes_select_separate_paths_and_scale(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            for mode, suffix in [('student', ''), ('demo', '_demo')]:
                (root / 'hw1_config.json').write_text(json.dumps(dict(mode=mode, scale=0.5)))
                cfg = IO.load_config(root)
                self.assertEqual(cfg['DATA'], str(root / ('data' + suffix)))
                self.assertEqual(cfg['RESULTS'], str(root / ('results' + suffix)))
                self.assertEqual(cfg['SCALE'], 0.5)
                self.assertEqual(cfg['DEMO_DATA'], str(root / 'data_demo'))
            (root / 'hw1_config.json').write_text(json.dumps(dict(
                mode='student', demo={'data': 'custom_demo', 'results': 'custom_results'})))
            self.assertEqual(IO.load_config(root)['DEMO_DATA'], str(root / 'custom_demo'))
            with self.assertRaises(FileNotFoundError):
                IO.load_config(root / 'missing')

    def test_config_rejects_bad_mode_scale_and_output_collision(self):
        invalid = [dict(mode='wrong'), dict(mode='student', scale=0),
                   dict(mode='student', scale=True),
                   dict(mode='demo', demo={'data': 'data_demo', 'results': 'results'})]
        with tempfile.TemporaryDirectory() as temp:
            for config in invalid:
                Path(temp, 'hw1_config.json').write_text(json.dumps(config))
                with self.subTest(config=config), self.assertRaises(ValueError):
                    IO.load_config(temp)

    def test_corner_archive_keeps_source_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            det = dict(names=['a.jpg'], imgpoints=[np.zeros((54, 2))], objpoints=[np.zeros((54, 3))],
                       image_size=(640, 480), scale=0.5, pattern=(9, 6), square=24.0,
                       failed=[], source_folder='/sample/data/setC')
            path = os.path.join(temp, 'corners.npz')
            Bd.save_corners(path, det)
            restored = Bd.load_corners(path)
            self.assertEqual(restored['source_folder'], det['source_folder'])
            self.assertEqual(restored['image_size'], det['image_size'])


class ClickToolRegressionTests(unittest.TestCase):
    def test_polyline_undo_reopens_completed_edges(self):
        clicker = CT.NotebookClicker(np.zeros((5, 5, 3)), kind='polylines', spec={'a': 2, 'b': 2})
        clicker.ax = object()
        clicker.fig = types.SimpleNamespace(canvas=types.SimpleNamespace(toolbar=None))
        clicker._redraw = lambda: None
        def event(button):
            clicker._on_click(types.SimpleNamespace(inaxes=clicker.ax, xdata=1, ydata=2, button=button))
        event(1); event(1); event(3)
        self.assertEqual([len(v) for v in clicker.polylines.values()], [1, 0])
        event(1); event(1); event(1); event(3)
        self.assertEqual([len(v) for v in clicker.polylines.values()], [2, 1])
        event(2); event(3)
        self.assertNotIn('b', clicker.closed)
        self.assertEqual(len(clicker.polylines['b']), 0)

    def test_cli_backend_falls_back_after_import_or_figure_failure(self):
        import matplotlib
        import matplotlib.pyplot as plt
        for error_at in ['import', 'figure']:
            imported = []
            def load_module(name):
                imported.append(name)
                if error_at == 'import' and name.endswith('backend_qtagg'):
                    raise ImportError('Qt is not installed')
                return object()
            figure_effects = [RuntimeError('Qt cannot create a window'), object()] if error_at == 'figure' else [object()]
            with self.subTest(error_at=error_at), patch.object(importlib, 'import_module', side_effect=load_module), \
                    patch.object(matplotlib, 'use'), patch.object(plt, 'figure', side_effect=figure_effects), \
                    patch.object(plt, 'close'):
                self.assertEqual(CT._select_interactive_backend(), 'TkAgg')
                self.assertTrue(imported[-1].endswith('backend_tkagg'))


class CheckContractRegressionTests(unittest.TestCase):
    def test_algorithm82_checker_supplies_homogeneous_vps(self):
        seen = []
        def supplied_algorithm(*vps):
            self.assertEqual([np.shape(v) for v in vps], [(3,), (3,), (3,)])
            self.assertTrue(all(v[2] == 1 for v in vps))
            seen.append(True)
            return S.K_TRUE.copy(), np.ones(3)
        with patch.object(CK, '_report'):
            CK.check_three_vps(lambda *v: S.K_TRUE[:2, 2], lambda *v: S.K_TRUE[0, 0], supplied_algorithm)
        self.assertEqual(seen, [True])

    def test_pitch_checker_supplies_homogeneous_vp(self):
        seen = []
        def supplied_algorithm(K, vp):
            self.assertEqual(np.shape(vp), (3,))
            self.assertEqual(vp[2], 1)
            seen.append(True)
            return 0.0, 0.0
        with patch.object(CK, '_report'):
            CK.check_pitch(supplied_algorithm)
        self.assertEqual(seen, [True])


if __name__ == '__main__':
    unittest.main()
