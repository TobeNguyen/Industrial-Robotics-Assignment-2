# Including libraries
import io
import os
import numpy as np
import cv2
from spatialmath import SE3

# Pillow is optional: fallback decoder (GIF, ICO, HEIC) and EXIF rotation
try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# Image formats accepted by load_image()
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp',
                        '.gif', '.ico', '.heic')

# Segmentation settings
MAX_IMAGE_SIDE = 1200            # longest side used for shape extraction (px)
MIN_CONTRAST = 15               # below this the image is treated as blank
MIN_TRANSPARENT_FRACTION = 0.01 # share of see-through pixels needed to trust alpha
MIN_FILL_RATIO = 0.3            # filled area / bounding box, below = outline still open
MIN_OBJECT_FRACTION = 0.01      # smallest region (of the image) counted as the object

# Multi-stroke settings
MIN_STROKE_FRACTION = 0.002     # smallest inner stroke, as a share of the object area
MAX_STROKES = 20                 # outer outline plus inner strokes, at most

# Inner stroke detection (local contrast) and polygon simplification
INNER_BLOCK_FRACTION = 0.08     # blur window for the local average, as a share of the image side
INNER_CONTRAST = 18             # grey-level difference from the local average that counts as a stroke
SIMPLIFY_EPSILON_FRAC = 0.0007  # default simplification, as a share of the stroke perimeter
SIMPLIFY_MIN_PX = 0.5           # never simplify finer than this (removes pixel staircase)
SIMPLIFY_MAX_PX = 1.5           # never simplify coarser than this (keeps big outlines accurate)

# Short messages about how the last image was processed (for the GUI log)
processing_notes = []

def _note(message):
    processing_notes.append(message)
    if len(processing_notes) > 50:
        del processing_notes[0]

# Returns a copy of the processing notes
def get_processing_notes():
    return list(processing_notes)

# Shape Model
class ShapeModel:
    def __init__(self, points, cx=0.0, cy=0.0, size=1.0, kind='complex'):
        self.kind = kind
        self.cx = cx
        self.cy = cy
        self.size = size               # overall half-extent, for raster bounds
        self.points = np.asarray(points, dtype=float)   # (N, 2) local coords

    # Boundary points in WORLD coordinates
    def boundary_points(self, n=360):
        pts = self.points
        # cumulative arc length, so resampling is even along the path rather
        # than bunched where the original contour had more points
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        total = arc[-1]
        if total == 0:
            return [(self.cx, self.cy)] * n
        targets = np.linspace(0, total, n)
        xs = np.interp(targets, arc, pts[:, 0])
        ys = np.interp(targets, arc, pts[:, 1])
        return [(self.cx + x, self.cy + y) for x, y in zip(xs, ys)]
    
    # Half-width of the shape at row offset dy from centre
    def half_width_at(self, dy):
        pts = self.points
        crossings = []
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            if (y0 <= dy < y1) or (y1 <= dy < y0):
                t = (dy - y0) / (y1 - y0)
                crossings.append(x0 + t * (x1 - x0))
        if len(crossings) < 2:
            return 0.0
        crossings.sort()
        # even-odd pairs give the filled spans at this row; return the
        # half-width of the outermost span, consistent with the basic
        # shapes' symmetric half_width_at() contract
        return max(abs(crossings[0]), abs(crossings[-1]))
    
# Build a ShapeModel for one of the three basic shapes
def generate_basic_shape(kind, size, n=360):
    from staubli_tx2_60_task import ShapeSpec
    spec = ShapeSpec(cx=0.0, cy=0.0, size=size, kind=kind)
    pts = np.array(spec.boundary_points(n))
    return ShapeModel(points=pts, cx=0.0, cy=0.0, size=size, kind=kind)

# Build a ShapeModel from a contour already extracted from an image
def create_shape_model(points, cx=0.0, cy=0.0, size=1.0):
    return ShapeModel(points=points, cx=cx, cy=cy, size=size, kind='complex')

# JPEG/Image processing
# Makes any decoded array an 8-bit image (16-bit scans, float TIFFs)
def _to_uint8(image):
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return (image // 257).astype(np.uint8)
    return cv2.normalize(image.astype(np.float32), None, 0, 255,
                        cv2.NORM_MINMAX).astype(np.uint8)

# Splits any image into a 3-channel BGR array and an alpha channel (None if opaque)
def _split_alpha(image):
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None
    channels = image.shape[2]
    if channels == 1:
        return cv2.cvtColor(np.ascontiguousarray(image[:, :, 0]), cv2.COLOR_GRAY2BGR), None
    if channels == 2:       # grayscale + transparency
        gray = np.ascontiguousarray(image[:, :, 0])
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), np.ascontiguousarray(image[:, :, 1])
    if channels == 4:
        return np.ascontiguousarray(image[:, :, :3]), np.ascontiguousarray(image[:, :, 3])
    return image, None

# Rotates the image to match its EXIF orientation tag (phone photos)
def _apply_exif_orientation(image, data):
    if Image is None:
        return image
    try:
        orientation = Image.open(io.BytesIO(data)).getexif().get(274, 1)
    except Exception:
        return image
    ops = {
        2: lambda a: a[:, ::-1],
        3: lambda a: a[::-1, ::-1],
        4: lambda a: a[::-1],
        5: lambda a: a.swapaxes(0, 1),
        6: lambda a: np.rot90(a, k=-1),
        7: lambda a: a[::-1, ::-1].swapaxes(0, 1),
        8: lambda a: np.rot90(a, k=1),
    }
    if orientation not in ops:
        return image
    _note(f'applied EXIF orientation {orientation}')
    return np.ascontiguousarray(ops[orientation](image))

# Fallback decoder for formats OpenCV cannot read
def _decode_with_pillow(data):
    if Image is None:
        return None
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
        has_alpha = 'A' in pil.getbands() or 'transparency' in pil.info
        pil = pil.convert('RGBA' if has_alpha else 'RGB')
        arr = np.array(pil)
    except Exception:
        return None
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA if has_alpha else cv2.COLOR_RGB2BGR)

# Reads an image from disk keeping colour and transparency (8-bit, EXIF-corrected)
def load_image_raw(path):
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f'load_image: unsupported format {ext!r}, use one of {SUPPORTED_EXTENSIONS}')
    # np.fromfile + imdecode also works with non-ASCII paths on Windows
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        raise ValueError(f'load_image: could not read image at {path!r}')
    if data.size == 0:
        raise ValueError(f'load_image: could not read image at {path!r}')
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        image = _decode_with_pillow(data)
        if image is None:
            raise ValueError(f'load_image: could not read image at {path!r}')
        _note('decoded with Pillow')
    else:
        image = _apply_exif_orientation(image, data)
    return _to_uint8(image)

# Reads an image from disk as a grayscale array. Accepts the formats listed in SUPPORTED_EXTENSIONS.
def load_image(path):
    image = load_image_raw(path)
    bgr, alpha = _split_alpha(image)
    # Transparent pixels (PNG/WebP) are composited onto a white background
    if alpha is not None:
        a = alpha[:, :, None] / 255.0
        bgr = (bgr * a + 255 * (1 - a)).astype(np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

# Shrinks very large images so the result is stable and fast
def limit_image_size(image, max_side=MAX_IMAGE_SIDE):
    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        return image
    _note(f'resized image from {w}x{h} to fit {max_side}px')
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                    interpolation=cv2.INTER_AREA)

def _odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1

def _ellipse(ksize):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))

def _touches_edge(contour, shape):
    x, y, w, h = cv2.boundingRect(contour)
    return x == 0 or y == 0 or x + w >= shape[1] or y + h >= shape[0]

# Closes small gaps in the outline, drops specks and fills the inside of every outer contour
def _fill_holes(mask, ksize, min_area):
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _ellipse(ksize))
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if hierarchy is None:
        return filled
    outer = [c for c, hv in zip(contours, hierarchy[0])
            if hv[3] == -1 and cv2.contourArea(c) >= min_area]
    # a frame or shadow glued to the image edge is ignored when a real object exists
    inner = [c for c in outer if not _touches_edge(c, mask.shape)]
    if any(cv2.contourArea(c) >= MIN_OBJECT_FRACTION * mask.size for c in inner):
        outer = inner
    cv2.drawContours(filled, outer, -1, 255, thickness=cv2.FILLED)
    return filled

# Keeps the largest group of nearby regions and joins its parts into one region
def _select_object(filled, gap):
    kernel = _ellipse(gap)
    count, labels = cv2.connectedComponents(cv2.dilate(filled, kernel), connectivity=8)
    best, best_area = None, 0
    for i in range(1, count):
        group = ((labels == i) & (filled > 0)).astype(np.uint8) * 255
        area = cv2.countNonZero(group)
        if area > best_area:
            best, best_area = group, area
    if best is None:
        return None
    joined = cv2.morphologyEx(best, cv2.MORPH_CLOSE, kernel)
    return _fill_holes(joined, 1, 0)

# Filled area divided by bounding-box area (low = outline still open)
def _fill_ratio(mask):
    x, y, w, h = cv2.boundingRect(mask)
    return cv2.countNonZero(mask) / float(max(w * h, 1))

# Separates object from background: returns the raw thresholded mask (255 = object pixels)
def _foreground_mask(image, blur_ksize=5):
    image = _to_uint8(image)
    bgr, alpha = _split_alpha(image)
    h, w = bgr.shape[:2]
    blurred = cv2.GaussianBlur(bgr, (blur_ksize, blur_ksize), 0)

    # 1. separate object from background
    if alpha is not None and np.mean(alpha < 250) >= MIN_TRANSPARENT_FRACTION:
        diff = cv2.GaussianBlur(alpha, (blur_ksize, blur_ksize), 0).astype(np.float32)
        _note('used transparency channel as background')
    else:
        border = max(1, min(h, w) // 100)
        edge = np.concatenate([
            blurred[:border].reshape(-1, 3), blurred[-border:].reshape(-1, 3),
            blurred[:, :border].reshape(-1, 3), blurred[:, -border:].reshape(-1, 3)])
        background = np.median(edge, axis=0)
        # distance over all 3 colour channels, not just grey level
        diff = np.linalg.norm(blurred.astype(np.float32) - background, axis=2)
        _note(f'background colour (BGR) {background.astype(int).tolist()}')
    if diff.max() < MIN_CONTRAST:
        raise ValueError('preprocess_image: no clear object found (image is blank)')
    diff8 = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 2. automatic threshold keeps the whole object area, not only its edges
    _, mask = cv2.threshold(diff8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

# Cleans the image up before contour extraction: returns a filled object mask (255 = object)
def preprocess_image(image, blur_ksize=5):
    mask = _foreground_mask(image, blur_ksize)
    h, w = mask.shape[:2]

    # 3-4. close outline gaps, fill, pick one object; grow the kernel if still open
    size = max(h, w)
    min_area = 1e-4 * h * w
    base = _odd(max(3, round(size * 0.01)))
    obj = None
    for scale in (1, 2, 4, 8):
        ksize = _odd(base * scale)
        filled = _fill_holes(mask, ksize, min_area)
        candidate = _select_object(filled, _odd(max(ksize, round(size * 0.03))))
        if candidate is None:
            continue
        obj = candidate
        if _fill_ratio(obj) >= MIN_FILL_RATIO:
            _note(f'closed outline with kernel {ksize}')
            break
    else:
        if obj is None:
            raise ValueError('preprocess_image: no clear object found in image')
        # last resort: convex hull of what was found
        hull = cv2.convexHull(cv2.findNonZero(obj))
        obj = np.zeros_like(obj)
        cv2.fillConvexPoly(obj, hull, 255)
        _note('outline stayed open, used convex hull')
    return obj

# Finds the outer contour of the preprocessed image and returns the largest one as an (N, 2) array of pixel coordinates
def extract_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('extract_contours: no contour found in image')
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(float)

# Reduce a dense pixel contour (tolerance is clamped so small strokes keep their shape and big outlines stay accurate)
def simplify_contour(points, epsilon_frac=SIMPLIFY_EPSILON_FRAC, max_px=SIMPLIFY_MAX_PX):
    pts = points.astype(np.float32)
    perimeter = cv2.arcLength(pts, closed=True)
    epsilon = min(max(epsilon_frac * perimeter, SIMPLIFY_MIN_PX), max_px)
    simplified = cv2.approxPolyDP(pts, epsilon, closed=True)
    return simplified.reshape(-1, 2).astype(float)

# True when segments p1-p2 and p3-p4 properly cross each other
def _segments_cross(p1, p2, p3, p4):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return d1 * d2 < 0 and d3 * d4 < 0

# True when a closed polygon does not cross itself
def is_simple_polygon(points):
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue        # these two segments share a vertex
            if _segments_cross(a, b, points[j], points[(j + 1) % n]):
                return False
    return True

# Area centroid of a closed polygon (the plain mean of the vertices is biased where the vertices are denser)
def _centroid(points):
    m = cv2.moments(np.asarray(points, dtype=np.float32))
    if abs(m['m00']) < 1e-9:
        return points.mean(axis=0)
    return np.array([m['m10'] / m['m00'], m['m01'] / m['m00']])

# Centres the contour on its own centroid
def normalise_path(points, target_size):
    centroid = _centroid(points)
    centred = points - centroid
    extent = np.max(np.abs(centred))
    if extent == 0:
        raise ValueError('normalise_path: contour has zero extent')
    scaled = centred * (target_size / extent)
    closed = np.vstack([scaled, scaled[:1]])
    return closed

# One-call pipeline: JPEG path -> ShapeModel
def shape_model_from_jpeg(path, target_size, simplify_epsilon=SIMPLIFY_EPSILON_FRAC):
    del processing_notes[:]
    image = limit_image_size(load_image_raw(path))
    binary = preprocess_image(image)
    raw_points = extract_contours(binary)
    simple_points = simplify_contour(raw_points, simplify_epsilon)
    # a self-crossing outline would give an invalid stamp and drawing path
    if not is_simple_polygon(simple_points):
        hull = cv2.convexHull(simple_points.astype(np.float32))
        simple_points = hull.reshape(-1, 2).astype(float)
        _note('contour self-intersects, used convex hull')
    if len(simple_points) < 3:
        raise ValueError('shape_model_from_jpeg: object too small to form a shape')
    local_points = normalise_path(simple_points, target_size)
    return create_shape_model(local_points, cx=0.0, cy=0.0, size=target_size)

# Multi-stroke drawing
# Finds separate detail regions (eyes, mouth...) inside the object, as a list of (N, 2) contours.
# A stroke is a pixel that differs from its local average AND from the object's own tone, on the
# same side (darker or lighter), so thin and faint lines survive without growing a halo around them.
# Regions glued to the outer outline are skipped because the outline is already the first stroke.
def extract_inner_contours(image, object_mask, blur_ksize=5):
    bgr, _ = _split_alpha(_to_uint8(image))
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    h, w = gray.shape
    block = _odd(max(11, round(max(h, w) * INNER_BLOCK_FRACTION)))
    local = cv2.GaussianBlur(gray, (block, block), 0)
    base = float(np.median(gray[object_mask > 0]))      # typical tone of the object surface
    g = gray.astype(np.float32)
    lo = local.astype(np.float32)
    dark = (g < lo - INNER_CONTRAST) & (g < base - INNER_CONTRAST / 2)
    light = (g > lo + INNER_CONTRAST) & (g > base + INNER_CONTRAST / 2)
    strokes = ((dark | light).astype(np.uint8)) * 255
    inside = cv2.bitwise_and(strokes, object_mask)
    # joins tiny gaps in a line so one stroke stays one region
    inside = cv2.morphologyEx(inside, cv2.MORPH_CLOSE, _ellipse(3))
    # thin band along the object boundary, any region touching it belongs to the outline
    edge = cv2.subtract(object_mask, cv2.erode(object_mask, _ellipse(5)))
    count, labels = cv2.connectedComponents(inside, connectivity=8)
    touching = set(np.unique(labels[edge > 0]).tolist())
    details = np.zeros_like(inside)
    for i in range(1, count):
        if i not in touching:
            details[labels == i] = 255
    contours, _ = cv2.findContours(details, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = MIN_STROKE_FRACTION * cv2.countNonZero(object_mask)
    kept = [c for c in contours if cv2.contourArea(c) >= min_area]
    kept.sort(key=cv2.contourArea, reverse=True)
    return [c.reshape(-1, 2).astype(float) for c in kept[:MAX_STROKES - 1]]

# Simplifies one contour and makes sure it does not cross itself.
# A crossing is first fixed with a finer tolerance; only the outer outline may fall back
# to the convex hull, an inner stroke is dropped instead of being distorted (None if too small)
def _clean_polygon(raw_points, simplify_epsilon, allow_hull=True):
    for scale in (1.0, 0.5, 0.25):
        points = simplify_contour(raw_points, simplify_epsilon * scale, SIMPLIFY_MAX_PX * scale)
        if is_simple_polygon(points):
            return points if len(points) >= 3 else None
    if not allow_hull:
        _note('stroke self-intersects, dropped')
        return None
    hull = cv2.convexHull(points.astype(np.float32))
    points = hull.reshape(-1, 2).astype(float)
    _note('stroke self-intersects, used convex hull')
    return points if len(points) >= 3 else None

# Centres all strokes on the FIRST (outer) stroke and scales them together, so they stay aligned
def normalise_strokes(strokes, target_size):
    outer = strokes[0]
    centroid = _centroid(outer)
    extent = np.max(np.abs(outer - centroid))
    if extent == 0:
        raise ValueError('normalise_strokes: contour has zero extent')
    out = []
    for s in strokes:
        scaled = (s - centroid) * (target_size / extent)
        out.append(np.vstack([scaled, scaled[:1]]))
    return out

# Keeps the first stroke first, then picks the nearest next stroke and starts it at its closest point
def order_strokes(strokes):
    if len(strokes) == 0:
        return []
    ordered = [strokes[0]]
    remaining = list(strokes[1:])
    pen = strokes[0][-1]
    while remaining:
        best_i, best_j, best_d = 0, 0, np.inf
        for i, s in enumerate(remaining):
            d = np.linalg.norm(s[:-1] - pen, axis=1)
            j = int(np.argmin(d))
            if d[j] < best_d:
                best_i, best_j, best_d = i, j, d[j]
        s = remaining.pop(best_i)
        pts = np.roll(s[:-1], -best_j, axis=0)
        closed = np.vstack([pts, pts[:1]])
        ordered.append(closed)
        pen = closed[-1]
    return ordered

# One-call pipeline: image path -> list of ShapeModel strokes (outer outline first, then inner details)
def strokes_from_jpeg(path, target_size, simplify_epsilon=SIMPLIFY_EPSILON_FRAC):
    del processing_notes[:]
    image = limit_image_size(load_image_raw(path))
    binary = preprocess_image(image)
    outer = _clean_polygon(extract_contours(binary), simplify_epsilon, allow_hull=True)
    if outer is None:
        raise ValueError('strokes_from_jpeg: object too small to form a shape')
    strokes = [outer]
    for raw in extract_inner_contours(image, binary):
        points = _clean_polygon(raw, simplify_epsilon, allow_hull=False)
        if points is not None:
            strokes.append(points)
    strokes = order_strokes(normalise_strokes(strokes, target_size))
    _note(f'{len(strokes)} stroke(s) extracted')
    return [create_shape_model(s, cx=0.0, cy=0.0, size=target_size) for s in strokes]

# RGB-D
# Locates the workpiece in an RGB frame and returns its pixel-space boundary region
def detect_workpiece(rgb_image):
    binary = preprocess_image(rgb_image)
    contour = extract_contours(binary)
    pts = contour.astype(np.float32)
    x, y, w, h = cv2.boundingRect(pts)
    rect = cv2.minAreaRect(pts)
    # wrapped to [-45, 45) so a squarish workpiece does not flip between 0 and 90 degrees
    angle_deg = ((rect[-1] + 45.0) % 90.0) - 45.0
    return {'bbox': (x, y, w, h), 'center': rect[0], 'angle_deg': angle_deg, 'contour': contour}

# Reads the depth value (metres) around a pixel: median of the valid values in a small window
def estimate_depth(depth_image, pixel_xy, window=5):
    px, py = int(round(pixel_xy[0])), int(round(pixel_xy[1]))
    if not (0 <= py < depth_image.shape[0] and 0 <= px < depth_image.shape[1]):
        raise ValueError(f'estimate_depth: pixel {pixel_xy} outside depth image bounds')
    r = window // 2
    patch = depth_image[max(py - r, 0):py + r + 1, max(px - r, 0):px + r + 1].astype(float)
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else 0.0

# Combines detect_workpiece() and estimate_depth() into the workpiece's pose
def estimate_workpiece_pose(rgb_image, depth_image, intrinsics):
    fx, fy, px0, py0 = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    det = detect_workpiece(rgb_image)
    centre_px = det['center']
    z = estimate_depth(depth_image, centre_px)
    x = (centre_px[0] - px0) * z / fx
    y = (centre_px[1] - py0) * z / fy
    theta = np.deg2rad(det['angle_deg'])
    return {'x': x, 'y': y, 'z': z, 'theta': theta}

# Coordinate Transformation
# Builds the camera-to-robot transfrom from a one-time hand-eye calibration
def camera_to_robot(camera_pose_in_robot_frame):
    return SE3(camera_pose_in_robot_frame) if not isinstance(camera_pose_in_robot_frame, SE3) \
        else camera_pose_in_robot_frame

# Converts an estimate_workpiece_pose() result into a robot-frame SE3
def workpiece_to_robot(workpiece_pose, cam_to_robot):
    T_cam_obj = SE3(workpiece_pose['x'], workpiece_pose['y'], workpiece_pose['z']) \
        * SE3.Rz(workpiece_pose['theta'])
    return cam_to_robot * T_cam_obj

# Applies a robot-frame SE3 correction to a list of local ShapeModel points
def transform_path(points_local, correction, z_top):
    out = []
    for x, y in points_local:
        p = correction * SE3(x, y, 0.0)
        out.append((p.t[0], p.t[1], z_top))
    return out

# Applies the same correction to every stroke (ShapeModel or (N, 2) array), one path per stroke
def transform_strokes(strokes, correction, z_top):
    return [transform_path(s.points if hasattr(s, 'points') else s, correction, z_top)
            for s in strokes]

# Visual feedback
# Error between where the workpiece was expected to be and where RGB-D actually found
def calculate_pose_error(target_pose, actual_pose):
    def as_xytheta(pose):
        if isinstance(pose, SE3):
            x, y = pose.t[0], pose.t[1]
            theta = np.arctan2(pose.R[1, 0], pose.R[0, 0])
            return x, y, theta
        return pose

    tx, ty, ttheta = as_xytheta(target_pose)
    ax, ay, atheta = as_xytheta(actual_pose)
    # wrapped to [-pi, pi) so a yaw near +-pi (camera looking down) does not give a false 2*pi error
    dtheta = (atheta - ttheta + np.pi) % (2 * np.pi) - np.pi
    return np.array([ax - tx, ay - ty, dtheta])

# Turns of a pose error into the one-shot correction transformation to apply to JPEG-derived path
def calculate_visual_correction(pose_error):
    dx, dy, dtheta = pose_error
    return SE3(dx, dy, 0.0) * SE3.Rz(dtheta)

# Applies a visual correction to every point of a JPEG-derived path in one
def apply_visual_correction(points_local, correction, z_top):
    return transform_path(points_local, correction, z_top)

# Sensor Status
# Reports whether the RGB-D camera is currently producing usable frames
def get_sensor_status(rgb_image, depth_image):
    rgb_ok = rgb_image is not None and rgb_image.size > 0
    depth_ok = depth_image is not None and depth_image.size > 0 and np.any(depth_image > 0)
    return {'rgb_ok': rgb_ok, 'depth_ok': depth_ok, 'ready': rgb_ok and depth_ok}

# Convenience wrapper: given a target pose and a fresh RGB-D frame pair,returns the workpiece's actual pose, the pose error against target, and the correction transform in one call
def get_workpiece_state(rgb_image, depth_image, intrinsics, cam_to_robot, target_pose):
    status = get_sensor_status(rgb_image, depth_image)
    if not status['ready']:
        return {'status': status, 'actual_pose': None, 'error': None, 'correction': None}

    workpiece_pose = estimate_workpiece_pose(rgb_image, depth_image, intrinsics)
    actual_pose = workpiece_to_robot(workpiece_pose, cam_to_robot)
    error = calculate_pose_error(target_pose, actual_pose)
    correction = calculate_visual_correction(error)
    return {'status': status, 'actual_pose': actual_pose, 'error': error, 'correction': correction}

# One visual servoing step between strokes: returns a correction only when the error is above the deadband,
# otherwise (or when the camera is not ready) returns an identity correction so the robot does not jitter
def visual_servo_step(rgb_image, depth_image, intrinsics, cam_to_robot, target_pose,
                    tol_pos=0.001, tol_theta=np.deg2rad(0.5)):
    state = get_workpiece_state(rgb_image, depth_image, intrinsics, cam_to_robot, target_pose)
    error = state['error']
    if error is None:
        return {'status': state['status'], 'error': None, 'correction': SE3(), 'applied': False}
    if np.hypot(error[0], error[1]) < tol_pos and abs(error[2]) < tol_theta:
        return {'status': state['status'], 'error': error, 'correction': SE3(), 'applied': False}
    return {'status': state['status'], 'error': error,
            'correction': state['correction'], 'applied': True}

# Measures the workpiece ONCE before drawing, from several clean frames (list of (rgb, depth)).
# The median over frames removes depth/detection noise; the returned correction is meant to be
# reused for EVERY stroke, so drawn lines can never disturb the measurement
def measure_workpiece_state(frames, intrinsics, cam_to_robot, target_pose,
                            min_frames=3, tol_pos=0.001, tol_theta=np.deg2rad(0.5)):
    poses = []
    for rgb_image, depth_image in frames:
        if not get_sensor_status(rgb_image, depth_image)['ready']:
            continue
        try:
            poses.append(estimate_workpiece_pose(rgb_image, depth_image, intrinsics))
        except ValueError:
            continue
    status = {'frames_used': len(poses), 'frames_total': len(frames),
            'ready': len(poses) >= min_frames}
    if not status['ready']:
        return {'status': status, 'actual_pose': None, 'error': None,
                'correction': SE3(), 'applied': False}

    median_pose = {k: float(np.median([p[k] for p in poses])) for k in ('x', 'y', 'z', 'theta')}
    actual_pose = workpiece_to_robot(median_pose, cam_to_robot)
    error = calculate_pose_error(target_pose, actual_pose)
    if np.hypot(error[0], error[1]) < tol_pos and abs(error[2]) < tol_theta:
        return {'status': status, 'actual_pose': actual_pose, 'error': error,
                'correction': SE3(), 'applied': False}
    return {'status': status, 'actual_pose': actual_pose, 'error': error,
            'correction': calculate_visual_correction(error), 'applied': True}