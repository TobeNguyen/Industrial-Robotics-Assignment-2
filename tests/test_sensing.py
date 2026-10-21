# Including libraries
import os
import tempfile
import unittest
import numpy as np
import cv2
from spatialmath import SE3

try:
    from PIL import Image
except ImportError:
    Image = None

import sensing

class TestShapeModel(unittest.TestCase):
    def _square_points(self, half=1.0):
        # closed square loop, local coords
        return np.array([
            [-half, -half], [half, -half], [half, half], [-half, half], [-half, -half],
        ])

    def test_boundary_points_closed_loop(self):
        model = sensing.ShapeModel(points=self._square_points(), cx=0.0, cy=0.0, size=1.0)
        pts = model.boundary_points(n=40)
        np.testing.assert_allclose(pts[0], pts[-1], atol=1e-6)

    def test_boundary_points_count_matches_n(self):
        model = sensing.ShapeModel(points=self._square_points(), cx=0.0, cy=0.0, size=1.0)
        pts = model.boundary_points(n=50)
        self.assertEqual(len(pts), 50)

    def test_boundary_points_shifted_by_centre(self):
        model = sensing.ShapeModel(points=self._square_points(), cx=5.0, cy=-2.0, size=1.0)
        pts = model.boundary_points(n=20)
        for x, y in pts:
            self.assertGreaterEqual(x, 5.0 - 1.0 - 1e-6)
            self.assertLessEqual(x, 5.0 + 1.0 + 1e-6)
            self.assertGreaterEqual(y, -2.0 - 1.0 - 1e-6)
            self.assertLessEqual(y, -2.0 + 1.0 + 1e-6)

    def test_half_width_at_centre_matches_square_half_side(self):
        model = sensing.ShapeModel(points=self._square_points(half=1.0), cx=0.0, cy=0.0, size=1.0)
        self.assertAlmostEqual(model.half_width_at(0.0), 1.0, places=6)

    def test_half_width_at_zero_outside_extent(self):
        model = sensing.ShapeModel(points=self._square_points(half=1.0), cx=0.0, cy=0.0, size=1.0)
        self.assertEqual(model.half_width_at(5.0), 0.0)

    def test_degenerate_contour_boundary_points_do_not_crash(self):
        pts = np.array([[0.0, 0.0], [0.0, 0.0]])
        model = sensing.ShapeModel(points=pts, cx=0.0, cy=0.0, size=0.0)
        out = model.boundary_points(n=10)
        self.assertEqual(len(out), 10)

class TestGenerateBasicShape(unittest.TestCase):
    def test_circle_shape_model_has_points_on_radius(self):
        model = sensing.generate_basic_shape('circle', size=0.03, n=36)
        for x, y in model.points if model.points.ndim == 2 else []:
            pass  # points are local already-offset; check via boundary instead
        pts = model.boundary_points(n=36)
        for x, y in pts:
            r = np.hypot(x - model.cx, y - model.cy)
            self.assertAlmostEqual(r, 0.03, delta=1e-6)

    def test_square_and_triangle_also_build(self):
        for kind in ['square', 'triangle']:
            model = sensing.generate_basic_shape(kind, size=0.02, n=30)
            self.assertEqual(model.kind, kind)
            self.assertGreater(len(model.boundary_points(10)), 0)

class TestCreateShapeModel(unittest.TestCase):
    def test_create_shape_model_sets_fields(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
        model = sensing.create_shape_model(pts, cx=1.0, cy=2.0, size=1.0)
        self.assertEqual(model.kind, 'complex')
        self.assertEqual(model.cx, 1.0)
        self.assertEqual(model.cy, 2.0)
        np.testing.assert_allclose(model.points, pts)

class TestLoadImage(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _square_image(self, size=200, square=80):
        img = np.zeros((size, size), dtype=np.uint8)
        start = (size - square) // 2
        img[start:start + square, start:start + square] = 255
        return img

    def _write(self, name, img):
        path = os.path.join(self.tmp_dir.name, name)
        self.assertTrue(cv2.imwrite(path, img))
        return path

    def test_load_image_reads_all_supported_formats(self):
        img = self._square_image()
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            path = self._write(f'square{ext}', img)
            loaded = sensing.load_image(path)
            self.assertEqual(loaded.shape, img.shape, f'wrong shape for {ext}')
            self.assertEqual(loaded.ndim, 2, f'not grayscale for {ext}')

    def test_load_image_accepts_uppercase_extension(self):
        path = self._write('square.PNG', self._square_image())
        loaded = sensing.load_image(path)
        self.assertEqual(loaded.ndim, 2)

    def test_load_image_rejects_unsupported_extension(self):
        path = os.path.join(self.tmp_dir.name, 'shape.txt')
        with open(path, 'wb') as f:
            f.write(b'not an image')
        with self.assertRaises(ValueError):
            sensing.load_image(path)

    def test_load_image_raises_on_missing_file(self):
        with self.assertRaises(ValueError):
            sensing.load_image(os.path.join(self.tmp_dir.name, 'missing.png'))

    def test_load_image_raises_on_corrupt_file(self):
        path = os.path.join(self.tmp_dir.name, 'corrupt.png')
        with open(path, 'wb') as f:
            f.write(b'this is not a png')
        with self.assertRaises(ValueError):
            sensing.load_image(path)

    def test_load_image_raises_on_empty_file(self):
        path = os.path.join(self.tmp_dir.name, 'empty.png')
        open(path, 'wb').close()
        with self.assertRaises(ValueError):
            sensing.load_image(path)

    def test_load_image_handles_non_ascii_path(self):
        path = self._write('hình_vuông.png', self._square_image())
        loaded = sensing.load_image(path)
        self.assertEqual(loaded.shape, (200, 200))

    def test_load_image_flattens_transparent_pixels_to_white(self):
        # BGRA image: fully transparent black background, opaque black square
        bgra = np.zeros((100, 100, 4), dtype=np.uint8)
        bgra[30:70, 30:70, 3] = 255
        path = self._write('transparent.png', bgra)
        loaded = sensing.load_image(path)
        self.assertEqual(loaded[5, 5], 255)     # transparent -> white
        self.assertEqual(loaded[50, 50], 0)     # opaque black stays black

    def test_load_image_raw_converts_16_bit_to_8_bit(self):
        img16 = np.zeros((200, 200), dtype=np.uint16)
        img16[60:140, 60:140] = 65535
        path = self._write('square16.png', img16)
        loaded = sensing.load_image_raw(path)
        self.assertEqual(loaded.dtype, np.uint8)
        self.assertEqual(loaded[100, 100], 255)
        self.assertEqual(loaded[5, 5], 0)

    @unittest.skipIf(Image is None, 'Pillow not installed')
    def test_load_image_raw_applies_exif_rotation(self):
        # 100 wide x 50 high, orientation 6 (rotate 90 CW) -> 50 wide x 100 high
        img = Image.fromarray(np.zeros((50, 100, 3), dtype=np.uint8))
        exif = Image.Exif()
        exif[274] = 6
        path = os.path.join(self.tmp_dir.name, 'rotated.jpg')
        img.save(path, exif=exif)
        loaded = sensing.load_image_raw(path)
        self.assertEqual(loaded.shape[:2], (100, 50))

    @unittest.skipIf(Image is None, 'Pillow not installed')
    def test_load_image_reads_gif_through_pillow_fallback(self):
        path = os.path.join(self.tmp_dir.name, 'square.gif')
        Image.fromarray(self._square_image()).save(path)
        loaded = sensing.load_image(path)
        self.assertEqual(loaded.shape, (200, 200))
        self.assertEqual(loaded.ndim, 2)

class TestImageProcessing(unittest.TestCase):
    def _synthetic_square_image(self, size=200, square=80):
        img = np.zeros((size, size), dtype=np.uint8)
        start = (size - square) // 2
        end = start + square
        img[start:end, start:end] = 255
        return img

    def test_preprocess_image_returns_binary(self):
        img = self._synthetic_square_image()
        binary = sensing.preprocess_image(img)
        self.assertEqual(binary.shape, img.shape)
        unique_vals = set(np.unique(binary).tolist())
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_extract_contours_finds_a_contour(self):
        img = self._synthetic_square_image()
        binary = sensing.preprocess_image(img)
        contour = sensing.extract_contours(binary)
        self.assertGreater(len(contour), 0)
        self.assertEqual(contour.shape[1], 2)

    def test_extract_contours_raises_on_blank_image(self):
        blank = np.zeros((50, 50), dtype=np.uint8)
        with self.assertRaises(ValueError):
            sensing.extract_contours(sensing.preprocess_image(blank))

    def test_simplify_contour_reduces_point_count(self):
        img = self._synthetic_square_image()
        binary = sensing.preprocess_image(img)
        contour = sensing.extract_contours(binary)
        simplified = sensing.simplify_contour(contour, epsilon_frac=0.01)
        self.assertLessEqual(len(simplified), len(contour))
        self.assertGreaterEqual(len(simplified), 3)

    def test_normalise_path_centres_and_scales(self):
        pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        normalised = sensing.normalise_path(pts, target_size=1.0)
        centroid = pts.mean(axis=0)
        # closed loop: last point repeats first
        np.testing.assert_allclose(normalised[0], normalised[-1], atol=1e-6)
        # max abs coordinate should equal target_size
        self.assertAlmostEqual(np.max(np.abs(normalised[:-1])), 1.0, places=6)

    def test_normalise_path_raises_on_zero_extent(self):
        pts = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
        with self.assertRaises(ValueError):
            sensing.normalise_path(pts, target_size=1.0)

    def test_shape_model_from_jpeg_end_to_end(self):
        img = self._synthetic_square_image()
        tmp_path = '/tmp/_sensing_test_square.png'
        cv2.imwrite(tmp_path, img)
        model = sensing.shape_model_from_jpeg(tmp_path, target_size=0.05)
        self.assertEqual(model.kind, 'complex')
        pts = model.boundary_points(n=40)
        self.assertEqual(len(pts), 40)

class TestObjectSegmentation(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _write(self, name, img):
        path = os.path.join(self.tmp_dir.name, name)
        self.assertTrue(cv2.imwrite(path, img))
        return path

    def _assert_square_area(self, mask, side=80):
        area = np.count_nonzero(mask)
        self.assertGreater(area, 0.9 * side * side)
        self.assertLess(area, 1.1 * side * side)

    def test_transparent_png_uses_alpha_as_background(self):
        bgra = np.zeros((200, 200, 4), dtype=np.uint8)
        bgra[60:140, 60:140] = 255
        path = self._write('alpha.png', bgra)
        model = sensing.shape_model_from_jpeg(path, target_size=0.05)
        self.assertEqual(model.kind, 'complex')
        self.assertTrue(any('transparency' in n for n in sensing.get_processing_notes()))

    def test_coloured_background(self):
        rgb = np.full((200, 200, 3), (200, 60, 60), dtype=np.uint8)
        rgb[60:140, 60:140] = (40, 200, 220)
        self._assert_square_area(sensing.preprocess_image(rgb))

    def test_dark_object_on_light_background(self):
        img = np.full((200, 200), 255, dtype=np.uint8)
        img[60:140, 60:140] = 0
        self._assert_square_area(sensing.preprocess_image(img))

    def test_white_object_on_black_background(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        img[60:140, 60:140] = 255
        self._assert_square_area(sensing.preprocess_image(img))

    def test_open_outline_is_filled(self):
        # thick circle outline with a gap, like a cartoon drawing
        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        cv2.ellipse(img, (150, 150), (60, 60), 0, 0, 340, (0, 0, 0), 6)
        mask = sensing.preprocess_image(img)
        self.assertGreater(np.count_nonzero(mask), 0.5 * np.pi * 60 ** 2)
        self.assertEqual(mask[150, 150], 255)

    def test_16_bit_image_end_to_end(self):
        img16 = np.zeros((200, 200), dtype=np.uint16)
        img16[60:140, 60:140] = 65535
        path = self._write('square16.png', img16)
        model = sensing.shape_model_from_jpeg(path, target_size=0.05)
        self.assertEqual(model.kind, 'complex')

    @unittest.skipIf(Image is None, 'Pillow not installed')
    def test_grayscale_with_transparency(self):
        la = np.zeros((200, 200, 2), dtype=np.uint8)
        la[:, :, 0] = 255                # transparent white background
        la[60:140, 60:140] = (0, 255)    # opaque black square
        path = os.path.join(self.tmp_dir.name, 'gray_alpha.png')
        Image.fromarray(la, 'LA').save(path)
        model = sensing.shape_model_from_jpeg(path, target_size=0.05)
        self.assertEqual(model.kind, 'complex')

    def test_noisy_image(self):
        rng = np.random.default_rng(0)
        img = np.zeros((200, 200), dtype=np.float32)
        img[60:140, 60:140] = 255
        noisy = np.clip(img + rng.normal(0, 20, img.shape), 0, 255).astype(np.uint8)
        self._assert_square_area(sensing.preprocess_image(noisy))

    def test_blank_images_raise_value_error(self):
        for value in (0, 128, 255):
            blank = np.full((100, 100), value, dtype=np.uint8)
            with self.assertRaises(ValueError):
                sensing.preprocess_image(blank)

    def test_region_glued_to_edge_is_ignored(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        img[75:125, 75:125] = 255        # object in the middle
        img[25:175, 0:25] = 255          # bigger strip touching the left edge
        mask = sensing.preprocess_image(img)
        self.assertEqual(mask[100, 100], 255)
        self.assertEqual(mask[100, 5], 0)

    def test_nearby_parts_are_joined(self):
        img = np.zeros((300, 300), dtype=np.uint8)
        img[100:140, 100:140] = 255
        img[100:140, 145:185] = 255      # 5 px gap, like a multi-part logo
        mask = sensing.preprocess_image(img)
        contour = sensing.extract_contours(mask)
        width = contour[:, 0].max() - contour[:, 0].min()
        self.assertGreaterEqual(width, 80)

    def test_limit_image_size_shrinks_large_images(self):
        big = np.zeros((1000, 1600, 3), dtype=np.uint8)
        small = sensing.limit_image_size(big, max_side=800)
        self.assertEqual(max(small.shape[:2]), 800)
        same = sensing.limit_image_size(np.zeros((100, 100), dtype=np.uint8), max_side=800)
        self.assertEqual(same.shape, (100, 100))

    def test_is_simple_polygon(self):
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        bowtie = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=float)
        self.assertTrue(sensing.is_simple_polygon(square))
        self.assertFalse(sensing.is_simple_polygon(bowtie))

class TestMultiStroke(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _face_image(self):
        # closed thick circle outline with two separate eyes inside, like a simple cartoon face
        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        cv2.circle(img, (150, 150), 100, (0, 0, 0), 8)
        cv2.circle(img, (115, 125), 12, (0, 0, 0), -1)
        cv2.circle(img, (185, 125), 12, (0, 0, 0), -1)
        return img

    def _write(self, name, img):
        path = os.path.join(self.tmp_dir.name, name)
        self.assertTrue(cv2.imwrite(path, img))
        return path

    def _square(self, x0, y0=0.0, side=1.0):
        # closed square loop starting at (x0, y0)
        return np.array([[x0, y0], [x0 + side, y0], [x0 + side, y0 + side],
                        [x0, y0 + side], [x0, y0]], dtype=float)

    def test_extract_inner_contours_finds_the_eyes(self):
        img = self._face_image()
        outer = sensing.preprocess_image(img)
        inner = sensing.extract_inner_contours(img, outer)
        self.assertEqual(len(inner), 2)

    def test_extract_inner_contours_ignores_solid_object(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        img[60:140, 60:140] = 255
        outer = sensing.preprocess_image(img)
        self.assertEqual(sensing.extract_inner_contours(img, outer), [])

    def test_strokes_from_jpeg_returns_outline_then_details(self):
        path = self._write('face.png', self._face_image())
        strokes = sensing.strokes_from_jpeg(path, target_size=0.05)
        self.assertEqual(len(strokes), 3)
        for s in strokes:
            self.assertEqual(s.kind, 'complex')
            np.testing.assert_allclose(s.points[0], s.points[-1], atol=1e-9)
        # the eyes stay inside the outline and share its scale
        self.assertAlmostEqual(np.max(np.abs(strokes[0].points)), 0.05, places=6)
        for s in strokes[1:]:
            self.assertLess(np.max(np.abs(s.points)), 0.05)

    def test_strokes_from_jpeg_outer_matches_shape_model(self):
        path = self._write('face.png', self._face_image())
        strokes = sensing.strokes_from_jpeg(path, target_size=0.05)
        model = sensing.shape_model_from_jpeg(path, target_size=0.05)
        np.testing.assert_allclose(strokes[0].points, model.points)

    def test_strokes_from_jpeg_single_stroke_for_solid_shape(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        img[60:140, 60:140] = 255
        path = self._write('solid.png', img)
        strokes = sensing.strokes_from_jpeg(path, target_size=0.05)
        self.assertEqual(len(strokes), 1)

    def test_normalise_strokes_shares_one_scale(self):
        outer = np.array([[-10, -10], [10, -10], [10, 10], [-10, 10]], dtype=float)
        inner = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        out = sensing.normalise_strokes([outer, inner], target_size=1.0)
        self.assertAlmostEqual(np.max(np.abs(out[0])), 1.0, places=6)
        self.assertAlmostEqual(np.max(np.abs(out[1])), 0.1, places=6)
        np.testing.assert_allclose(out[1][0], out[1][-1], atol=1e-9)

    def test_order_strokes_keeps_first_and_picks_nearest(self):
        a = self._square(0.0)
        b = self._square(10.0)
        c = self._square(2.0)
        ordered = sensing.order_strokes([a, b, c])
        self.assertEqual(len(ordered), 3)
        np.testing.assert_allclose(ordered[0], a)
        # nearest stroke to a's end point (0, 0) is c, entered at its closest corner
        np.testing.assert_allclose(ordered[1][0], [2.0, 0.0])
        np.testing.assert_allclose(ordered[2][0], [10.0, 0.0])
        for s in ordered:
            np.testing.assert_allclose(s[0], s[-1], atol=1e-9)

    def test_order_strokes_empty_list(self):
        self.assertEqual(sensing.order_strokes([]), [])

    def test_transform_strokes_one_path_per_stroke(self):
        s1 = sensing.create_shape_model(self._square(0.0), size=1.0)
        s2 = self._square(2.0)       # plain arrays are accepted too
        correction = SE3(0.1, 0.2, 0.0)
        paths = sensing.transform_strokes([s1, s2], correction, z_top=0.35)
        self.assertEqual(len(paths), 2)
        self.assertEqual(len(paths[0]), len(s1.points))
        np.testing.assert_allclose(paths[0][0], [0.1, 0.2, 0.35], atol=1e-9)
        np.testing.assert_allclose(paths[1][0], [2.1, 0.2, 0.35], atol=1e-9)

class TestRGBD(unittest.TestCase):
    def _synthetic_rgb_and_contour(self, size=200, square=80, angle=0):
        gray = np.zeros((size, size), dtype=np.uint8)
        start = (size - square) // 2
        end = start + square
        gray[start:end, start:end] = 255
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return rgb

    def test_detect_workpiece_returns_bbox_and_angle(self):
        rgb = self._synthetic_rgb_and_contour()
        result = sensing.detect_workpiece(rgb)
        self.assertIn('bbox', result)
        self.assertIn('angle_deg', result)
        x, y, w, h = result['bbox']
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)

    def test_estimate_depth_reads_correct_pixel(self):
        depth = np.zeros((10, 10), dtype=float)
        depth[5, 3] = 0.42
        value = sensing.estimate_depth(depth, pixel_xy=(3, 5))
        self.assertAlmostEqual(value, 0.42)

    def test_estimate_depth_raises_out_of_bounds(self):
        depth = np.zeros((10, 10), dtype=float)
        with self.assertRaises(ValueError):
            sensing.estimate_depth(depth, pixel_xy=(50, 50))

    def test_estimate_workpiece_pose_returns_xyz_theta(self):
        rgb = self._synthetic_rgb_and_contour()
        depth = np.full((200, 200), 0.5, dtype=float)
        intrinsics = {'fx': 500.0, 'fy': 500.0, 'cx': 100.0, 'cy': 100.0}
        pose = sensing.estimate_workpiece_pose(rgb, depth, intrinsics)
        for key in ('x', 'y', 'z', 'theta'):
            self.assertIn(key, pose)
        self.assertAlmostEqual(pose['z'], 0.5, places=6)

class TestCoordinateTransform(unittest.TestCase):
    def test_camera_to_robot_accepts_se3(self):
        T = SE3(0.1, 0.2, 0.3)
        result = sensing.camera_to_robot(T)
        self.assertIsInstance(result, SE3)
        np.testing.assert_allclose(result.t, T.t)

    def test_camera_to_robot_accepts_raw_matrix(self):
        T = SE3(0.1, 0.2, 0.3)
        result = sensing.camera_to_robot(T.A)
        np.testing.assert_allclose(result.t, T.t, atol=1e-9)

    def test_workpiece_to_robot_combines_transforms(self):
        cam_to_robot = SE3(1.0, 0.0, 0.0)
        workpiece_pose = {'x': 0.5, 'y': 0.0, 'z': 0.3, 'theta': 0.0}
        result = sensing.workpiece_to_robot(workpiece_pose, cam_to_robot)
        np.testing.assert_allclose(result.t, [1.5, 0.0, 0.3], atol=1e-9)

    def test_transform_path_applies_correction_to_every_point(self):
        points_local = [(0.0, 0.0), (0.01, 0.0), (0.0, 0.01)]
        correction = SE3(0.1, 0.2, 0.0)
        out = sensing.transform_path(points_local, correction, z_top=0.35)
        self.assertEqual(len(out), 3)
        np.testing.assert_allclose(out[0], [0.1, 0.2, 0.35], atol=1e-9)
        np.testing.assert_allclose(out[1], [0.11, 0.2, 0.35], atol=1e-9)

    def test_transform_path_applies_rotation(self):
        points_local = [(1.0, 0.0)]
        correction = SE3.Rz(np.pi / 2)
        out = sensing.transform_path(points_local, correction, z_top=0.0)
        x, y, z = out[0]
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 1.0, places=6)

class TestVisualFeedback(unittest.TestCase):
    def test_calculate_pose_error_with_tuples(self):
        target = (0.400, 0.200, 0.0)
        actual = (0.407, 0.196, np.deg2rad(3))
        error = sensing.calculate_pose_error(target, actual)
        self.assertAlmostEqual(error[0], 0.007, places=6)
        self.assertAlmostEqual(error[1], -0.004, places=6)
        self.assertAlmostEqual(error[2], np.deg2rad(3), places=6)

    def test_calculate_pose_error_with_se3(self):
        target = SE3(0.4, 0.2, 0.0)
        actual = SE3(0.407, 0.196, 0.0) * SE3.Rz(np.deg2rad(3))
        error = sensing.calculate_pose_error(target, actual)
        self.assertAlmostEqual(error[0], 0.007, places=6)
        self.assertAlmostEqual(error[1], -0.004, places=6)

    def test_calculate_visual_correction_builds_se3(self):
        error = np.array([0.005, -0.003, np.deg2rad(2)])
        correction = sensing.calculate_visual_correction(error)
        self.assertIsInstance(correction, SE3)
        np.testing.assert_allclose(correction.t[:2], [0.005, -0.003], atol=1e-9)

    def test_apply_visual_correction_matches_transform_path(self):
        points_local = [(0.0, 0.0), (0.01, 0.0)]
        error = np.array([0.1, 0.2, 0.0])
        correction = sensing.calculate_visual_correction(error)
        out = sensing.apply_visual_correction(points_local, correction, z_top=0.35)
        expected = sensing.transform_path(points_local, correction, z_top=0.35)
        np.testing.assert_allclose(out, expected)

    def test_zero_error_gives_identity_correction(self):
        error = np.array([0.0, 0.0, 0.0])
        correction = sensing.calculate_visual_correction(error)
        np.testing.assert_allclose(correction.A, np.eye(4), atol=1e-9)

class TestSensorStatus(unittest.TestCase):
    def _ready_frames(self):
        size = 200
        gray = np.zeros((size, size), dtype=np.uint8)
        gray[60:140, 60:140] = 255
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        depth = np.full((size, size), 0.5, dtype=float)
        intrinsics = {'fx': 500.0, 'fy': 500.0, 'cx': 100.0, 'cy': 100.0}
        return rgb, depth, intrinsics

    def test_get_sensor_status_ready_when_both_present(self):
        rgb = np.ones((10, 10, 3), dtype=np.uint8)
        depth = np.ones((10, 10), dtype=float) * 0.5
        status = sensing.get_sensor_status(rgb, depth)
        self.assertTrue(status['rgb_ok'])
        self.assertTrue(status['depth_ok'])
        self.assertTrue(status['ready'])

    def test_get_sensor_status_not_ready_with_none_rgb(self):
        depth = np.ones((10, 10), dtype=float) * 0.5
        status = sensing.get_sensor_status(None, depth)
        self.assertFalse(status['rgb_ok'])
        self.assertFalse(status['ready'])

    def test_get_sensor_status_not_ready_with_all_zero_depth(self):
        rgb = np.ones((10, 10, 3), dtype=np.uint8)
        depth = np.zeros((10, 10), dtype=float)
        status = sensing.get_sensor_status(rgb, depth)
        self.assertFalse(status['depth_ok'])
        self.assertFalse(status['ready'])

    def test_get_workpiece_state_returns_none_when_not_ready(self):
        depth = np.zeros((10, 10), dtype=float)
        result = sensing.get_workpiece_state(
            rgb_image=None, depth_image=depth, intrinsics={}, cam_to_robot=SE3(),
            target_pose=(0, 0, 0))
        self.assertIsNone(result['actual_pose'])
        self.assertIsNone(result['correction'])
        self.assertFalse(result['status']['ready'])

    def test_get_workpiece_state_returns_correction_when_ready(self):
        size = 200
        gray = np.zeros((size, size), dtype=np.uint8)
        gray[60:140, 60:140] = 255
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        depth = np.full((size, size), 0.5, dtype=float)
        intrinsics = {'fx': 500.0, 'fy': 500.0, 'cx': 100.0, 'cy': 100.0}
        target_pose = (0.0, 0.0, 0.0)

        result = sensing.get_workpiece_state(
            rgb, depth, intrinsics, cam_to_robot=SE3(), target_pose=target_pose)
        self.assertTrue(result['status']['ready'])
        self.assertIsNotNone(result['actual_pose'])
        self.assertIsNotNone(result['correction'])
        self.assertEqual(len(result['error']), 3)

    def test_visual_servo_step_identity_when_not_ready(self):
        depth = np.zeros((10, 10), dtype=float)
        result = sensing.visual_servo_step(
            None, depth, intrinsics={}, cam_to_robot=SE3(), target_pose=(0, 0, 0))
        self.assertFalse(result['applied'])
        self.assertIsNone(result['error'])
        np.testing.assert_allclose(result['correction'].A, np.eye(4), atol=1e-9)

    def test_visual_servo_step_ignores_error_inside_deadband(self):
        rgb, depth, intrinsics = self._ready_frames()
        state = sensing.get_workpiece_state(
            rgb, depth, intrinsics, cam_to_robot=SE3(), target_pose=(0.0, 0.0, 0.0))
        # target equals where the workpiece actually is -> zero error
        result = sensing.visual_servo_step(
            rgb, depth, intrinsics, cam_to_robot=SE3(), target_pose=state['actual_pose'])
        self.assertFalse(result['applied'])
        np.testing.assert_allclose(result['correction'].A, np.eye(4), atol=1e-9)

    def test_visual_servo_step_applies_large_error(self):
        rgb, depth, intrinsics = self._ready_frames()
        result = sensing.visual_servo_step(
            rgb, depth, intrinsics, cam_to_robot=SE3(), target_pose=(0.1, 0.0, 0.0))
        self.assertTrue(result['applied'])
        self.assertAlmostEqual(result['error'][0], -0.1, places=2)
        self.assertIsInstance(result['correction'], SE3)

def _dot_image(n_cols, n_rows, radius=16):
    img = np.full((600, 600, 3), 255, dtype=np.uint8)
    cv2.circle(img, (300, 300), 260, (0, 0, 0), 8)
    for r in range(n_rows):
        for c in range(n_cols):
            cv2.circle(img, (130 + c * 68, 150 + r * 75), radius, (0, 0, 0), -1)
    return img

class TestSimplifyAndContours(unittest.TestCase):
    def _square_image(self, size=200, square=80):
        img = np.zeros((size, size), dtype=np.uint8)
        s = (size - square) // 2
        img[s:s + square, s:s + square] = 255
        return img

    def test_default_max_image_side_is_1200(self):
        self.assertEqual(sensing.MAX_IMAGE_SIDE, 1200)
        big = np.zeros((1000, 1300, 3), dtype=np.uint8)
        self.assertEqual(max(sensing.limit_image_size(big).shape[:2]), 1200)

    def test_extract_contours_keeps_every_boundary_point(self):
        binary = sensing.preprocess_image(self._square_image())
        contour = sensing.extract_contours(binary)
        # an 80x80 square has ~316 boundary pixels; CHAIN_APPROX_SIMPLE would give only 4
        self.assertGreater(len(contour), 250)

    def test_simplify_contour_never_coarser_than_max_px(self):
        img = np.zeros((700, 700), dtype=np.uint8)
        cv2.circle(img, (350, 350), 300, 255, -1)
        contour = sensing.extract_contours(sensing.preprocess_image(img))
        # a huge epsilon_frac would collapse the circle to a few points without the clamp
        simplified = sensing.simplify_contour(contour, epsilon_frac=0.5)
        self.assertGreater(len(simplified), 20)
        radii = np.hypot(simplified[:, 0] - 350, simplified[:, 1] - 350)
        self.assertLess(np.max(np.abs(radii - 300)), 4.0)

    def test_simplify_contour_keeps_small_stroke_shape(self):
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(img, (100, 100), 15, 255, -1)
        contour = sensing.extract_contours(img)
        simplified = sensing.simplify_contour(contour)
        self.assertGreater(len(simplified), 8)

class TestInnerStrokes(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _write(self, name, img):
        path = os.path.join(self.tmp_dir.name, name)
        self.assertTrue(cv2.imwrite(path, img))
        return path

    def _face(self, ring=(0, 0, 0), eye=(0, 0, 0)):
        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        cv2.circle(img, (150, 150), 100, ring, 8)
        cv2.circle(img, (115, 125), 12, eye, -1)
        cv2.circle(img, (185, 125), 12, eye, -1)
        return img

    def test_inner_contours_keep_true_size_of_details(self):
        # no halo: the contour must stay close to the real eye (radius 12 -> area ~452)
        img = self._face()
        inner = sensing.extract_inner_contours(img, sensing.preprocess_image(img))
        self.assertEqual(len(inner), 2)
        for c in inner:
            area = cv2.contourArea(c.astype(np.float32))
            self.assertGreater(area, 0.7 * np.pi * 12 ** 2)
            self.assertLess(area, 1.2 * np.pi * 12 ** 2)

    def test_inner_contours_thin_line_stays_thin(self):
        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        cv2.circle(img, (150, 150), 100, (0, 0, 0), 8)
        cv2.line(img, (110, 170), (190, 170), (0, 0, 0), 3)
        inner = sensing.extract_inner_contours(img, sensing.preprocess_image(img))
        self.assertEqual(len(inner), 1)
        x, y, w, h = cv2.boundingRect(inner[0].astype(np.float32))
        self.assertLessEqual(h, 8)           # line is 3 px thick
        self.assertGreaterEqual(w, 75)       # and about 80 px long

    def test_inner_contours_find_faint_details(self):
        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        cv2.circle(img, (150, 150), 100, (160, 160, 160), -1)
        cv2.circle(img, (115, 125), 12, (60, 60, 60), -1)
        cv2.circle(img, (185, 125), 12, (60, 60, 60), -1)
        inner = sensing.extract_inner_contours(img, sensing.preprocess_image(img))
        self.assertEqual(len(inner), 2)

    def test_inner_contours_find_light_details_on_dark_object(self):
        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        cv2.circle(img, (150, 150), 100, (40, 40, 40), -1)
        cv2.circle(img, (115, 125), 12, (255, 255, 255), -1)
        cv2.circle(img, (185, 125), 12, (255, 255, 255), -1)
        inner = sensing.extract_inner_contours(img, sensing.preprocess_image(img))
        self.assertEqual(len(inner), 2)

    def test_inner_contours_skip_stroke_touching_the_outline(self):
        img = self._face()
        cv2.line(img, (150, 50), (150, 110), (0, 0, 0), 4)   # starts on the ring
        inner = sensing.extract_inner_contours(img, sensing.preprocess_image(img))
        self.assertEqual(len(inner), 2)      # only the eyes, the line belongs to the outline

    def test_inner_contours_respect_max_strokes(self):
        img = _dot_image(6, 5)               # 30 separate dots
        inner = sensing.extract_inner_contours(img, sensing.preprocess_image(img))
        self.assertEqual(len(inner), sensing.MAX_STROKES - 1)

    def test_inner_contours_many_strokes_all_found_below_the_cap(self):
        img = _dot_image(3, 3)               # 9 separate dots
        inner = sensing.extract_inner_contours(img, sensing.preprocess_image(img))
        self.assertEqual(len(inner), 9)

    def test_clean_polygon_returns_simple_polygon(self):
        square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        out = sensing._clean_polygon(square, sensing.SIMPLIFY_EPSILON_FRAC, allow_hull=False)
        self.assertIsNotNone(out)
        self.assertTrue(sensing.is_simple_polygon(out))

    def test_clean_polygon_drops_self_crossing_inner_stroke(self):
        bowtie = np.array([[0, 0], [100, 100], [100, 0], [0, 100]], dtype=float)
        self.assertIsNone(sensing._clean_polygon(bowtie, sensing.SIMPLIFY_EPSILON_FRAC, allow_hull=False))

    def test_clean_polygon_outer_falls_back_to_convex_hull(self):
        bowtie = np.array([[0, 0], [100, 100], [100, 0], [0, 100]], dtype=float)
        out = sensing._clean_polygon(bowtie, sensing.SIMPLIFY_EPSILON_FRAC, allow_hull=True)
        self.assertIsNotNone(out)
        self.assertTrue(sensing.is_simple_polygon(out))
        self.assertEqual(len(out), 4)

    def test_clean_polygon_returns_none_for_tiny_contour(self):
        line = np.array([[0, 0], [1, 1]], dtype=float)
        self.assertIsNone(sensing._clean_polygon(line, sensing.SIMPLIFY_EPSILON_FRAC, allow_hull=True))

    def test_strokes_from_jpeg_keeps_all_strokes_in_place(self):
        # 3 dots on a row inside a ring: 1 outline + 3 details, positions must survive the pipeline
        img = np.full((300, 300, 3), 255, dtype=np.uint8)
        cv2.circle(img, (150, 150), 100, (0, 0, 0), 8)
        for x in (100, 150, 200):
            cv2.circle(img, (x, 150), 12, (0, 0, 0), -1)
        path = self._write('dots.png', img)
        strokes = sensing.strokes_from_jpeg(path, target_size=0.05)
        self.assertEqual(len(strokes), 4)
        centres = sorted(sensing._centroid(s.points)[0] for s in strokes[1:])
        extent = np.max(np.abs(strokes[0].points))
        self.assertAlmostEqual(extent, 0.05, places=6)
        scale = 0.05 / 104.0                 # outer radius incl. line width ~104 px
        for got, px in zip(centres, (100, 150, 200)):
            self.assertAlmostEqual(got, (px - 150) * scale, delta=0.0005)

    def test_strokes_from_jpeg_busy_image_is_deterministic(self):
        path = self._write('busy.png', _dot_image(4, 4))
        a = sensing.strokes_from_jpeg(path, target_size=0.05)
        b = sensing.strokes_from_jpeg(path, target_size=0.05)
        self.assertEqual(len(a), 17)
        for s, t in zip(a, b):
            np.testing.assert_allclose(s.points, t.points)

class TestRGBDFiltering(unittest.TestCase):
    def _rgb(self, x0=60, y0=60, x1=140, y1=140, size=200):
        gray = np.zeros((size, size), dtype=np.uint8)
        gray[y0:y1, x0:x1] = 255
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def test_estimate_depth_uses_median_of_window(self):
        depth = np.full((20, 20), 0.5, dtype=float)
        depth[10, 10] = 0.0                  # hole at the centre pixel
        depth[9, 9] = 3.0                    # outlier next to it
        self.assertAlmostEqual(sensing.estimate_depth(depth, (10, 10)), 0.5, places=6)

    def test_estimate_depth_all_invalid_returns_zero(self):
        depth = np.zeros((20, 20), dtype=float)
        self.assertEqual(sensing.estimate_depth(depth, (10, 10)), 0.0)

    def test_estimate_depth_near_image_border(self):
        depth = np.full((10, 10), 0.4, dtype=float)
        self.assertAlmostEqual(sensing.estimate_depth(depth, (0, 0)), 0.4, places=6)

    def test_detect_workpiece_returns_rect_centre(self):
        result = sensing.detect_workpiece(self._rgb())
        self.assertIn('center', result)
        cx, cy = result['center']
        self.assertAlmostEqual(cx, 99.5, delta=1.5)
        self.assertAlmostEqual(cy, 99.5, delta=1.5)

    def test_detect_workpiece_angle_is_wrapped(self):
        # axis-aligned square must read ~0 deg, never +-90
        self.assertAlmostEqual(sensing.detect_workpiece(self._rgb())['angle_deg'], 0.0, delta=1.0)
        for deg in (10, 30, 80, 100, 170):
            gray = np.zeros((300, 300), dtype=np.uint8)
            box = cv2.boxPoints(((150, 150), (100, 100), deg)).astype(np.int32)
            cv2.fillPoly(gray, [box], 255)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            angle = sensing.detect_workpiece(rgb)['angle_deg']
            self.assertGreaterEqual(angle, -45.0)
            self.assertLess(angle, 45.0)
            expected = ((deg + 45.0) % 90.0) - 45.0
            self.assertAlmostEqual(angle, expected, delta=2.0, msg=f'deg={deg}')

    def test_estimate_workpiece_pose_uses_object_centre(self):
        rgb = self._rgb(x0=120, y0=60, x1=180, y1=120)       # off-centre workpiece
        depth = np.full((200, 200), 0.5, dtype=float)
        intr = {'fx': 500.0, 'fy': 500.0, 'cx': 100.0, 'cy': 100.0}
        pose = sensing.estimate_workpiece_pose(rgb, depth, intr)
        self.assertAlmostEqual(pose['x'], (149.5 - 100.0) * 0.5 / 500.0, delta=0.002)
        self.assertAlmostEqual(pose['y'], (89.5 - 100.0) * 0.5 / 500.0, delta=0.002)
        self.assertAlmostEqual(pose['z'], 0.5, places=6)

class TestMeasureWorkpieceState(unittest.TestCase):
    def setUp(self):
        gray = np.zeros((200, 200), dtype=np.uint8)
        gray[60:140, 60:140] = 255
        self.rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self.depth = np.full((200, 200), 0.5, dtype=float)
        self.intr = {'fx': 500.0, 'fy': 500.0, 'cx': 100.0, 'cy': 100.0}

    def _frames(self, n=5):
        return [(self.rgb.copy(), self.depth.copy()) for _ in range(n)]

    def test_ready_with_enough_good_frames(self):
        result = sensing.measure_workpiece_state(
            self._frames(5), self.intr, SE3(), target_pose=(0.0, 0.0, 0.0))
        self.assertTrue(result['status']['ready'])
        self.assertEqual(result['status']['frames_used'], 5)
        self.assertEqual(result['status']['frames_total'], 5)
        self.assertIsNotNone(result['actual_pose'])
        self.assertEqual(len(result['error']), 3)

    def test_identity_when_too_few_frames(self):
        result = sensing.measure_workpiece_state(
            self._frames(2), self.intr, SE3(), target_pose=(0.0, 0.0, 0.0), min_frames=3)
        self.assertFalse(result['status']['ready'])
        self.assertFalse(result['applied'])
        self.assertIsNone(result['error'])
        np.testing.assert_allclose(result['correction'].A, np.eye(4), atol=1e-9)

    def test_identity_for_empty_frame_list(self):
        result = sensing.measure_workpiece_state([], self.intr, SE3(), (0.0, 0.0, 0.0))
        self.assertFalse(result['status']['ready'])
        self.assertEqual(result['status']['frames_used'], 0)

    def test_bad_frames_are_skipped(self):
        frames = self._frames(3)
        frames.append((None, self.depth))                                   # camera dropout
        frames.append((self.rgb, np.zeros((200, 200))))                     # no depth
        frames.append((np.zeros((200, 200, 3), dtype=np.uint8), self.depth))  # blank image
        result = sensing.measure_workpiece_state(frames, self.intr, SE3(), (0.0, 0.0, 0.0))
        self.assertEqual(result['status']['frames_total'], 6)
        self.assertEqual(result['status']['frames_used'], 3)
        self.assertTrue(result['status']['ready'])

    def test_median_rejects_one_outlier_frame(self):
        frames = self._frames(4)
        bad_depth = np.full((200, 200), 5.0, dtype=float)
        frames.append((self.rgb.copy(), bad_depth))
        result = sensing.measure_workpiece_state(frames, self.intr, SE3(), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(result['actual_pose'].t[2], 0.5, places=6)

    def test_no_correction_inside_deadband(self):
        first = sensing.measure_workpiece_state(self._frames(), self.intr, SE3(), (0.0, 0.0, 0.0))
        again = sensing.measure_workpiece_state(
            self._frames(), self.intr, SE3(), target_pose=first['actual_pose'])
        self.assertFalse(again['applied'])
        np.testing.assert_allclose(again['correction'].A, np.eye(4), atol=1e-9)

    def test_correction_applied_for_large_error(self):
        result = sensing.measure_workpiece_state(
            self._frames(), self.intr, SE3(), target_pose=(0.1, 0.0, 0.0))
        self.assertTrue(result['applied'])
        self.assertAlmostEqual(result['error'][0], -0.1, places=2)
        self.assertIsInstance(result['correction'], SE3)

    def test_one_correction_moves_every_stroke_the_same_way(self):
        # the point of measuring once: the same SE3 is reused for all strokes
        result = sensing.measure_workpiece_state(
            self._frames(), self.intr, SE3(), target_pose=(0.1, 0.0, 0.0))
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
        paths = sensing.transform_strokes([square, square * 0.5], result['correction'], z_top=0.35)
        shift_a = np.array(paths[0][0][:2]) - square[0]
        shift_b = np.array(paths[1][0][:2]) - (square * 0.5)[0]
        np.testing.assert_allclose(shift_a, shift_b, atol=1e-9)

class TestWristCameraFrame(unittest.TestCase):
    # Camera 1 m above the origin looking straight down: z along world -z, x along world -x (like the Staubli wrist camera)
    def _cam(self):
        return SE3(0.0, 0.0, 1.0) * SE3.Ry(np.pi)

    def test_workpiece_to_robot_for_a_downward_camera(self):
        pose = {'x': 0.01, 'y': 0.02, 'z': 0.2, 'theta': 0.0}
        result = sensing.workpiece_to_robot(pose, self._cam())
        np.testing.assert_allclose(result.t, [-0.01, 0.02, 0.8], atol=1e-9)

    def test_image_rotation_becomes_the_opposite_world_yaw(self):
        ref = sensing.workpiece_to_robot({'x': 0.0, 'y': 0.0, 'z': 0.2, 'theta': 0.0}, self._cam())
        for deg in (10, -10):
            act = sensing.workpiece_to_robot(
                {'x': 0.0, 'y': 0.0, 'z': 0.2, 'theta': np.deg2rad(deg)}, self._cam())
            error = sensing.calculate_pose_error(ref, act)
            self.assertAlmostEqual(error[2], -np.deg2rad(deg), places=6, msg=f'deg={deg}')

    def test_calculate_pose_error_wraps_across_pi(self):
        error = sensing.calculate_pose_error((0.0, 0.0, np.deg2rad(179)), (0.0, 0.0, np.deg2rad(-179)))
        self.assertAlmostEqual(error[2], np.deg2rad(2), places=6)

if __name__ == "__main__":
    unittest.main()