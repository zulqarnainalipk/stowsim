"""Orientation table, container mesh generation and small geometry helpers."""
import math
import numpy as np


ORNS = [
    [0, 0, 0],                 # 0: dimensions [l, w, h] (unchanged)
    [math.pi/2, 0, 0],         # 1: dimensions [l, h, w] (rotated 90 deg about X)
    [0, math.pi/2, 0],         # 2: dimensions [h, w, l] (rotated 90 deg about Y)
    [0, 0, math.pi/2],         # 3: dimensions [w, l, h] (rotated 90 deg about Z)
    [0, math.pi/2, math.pi/2], # 4: dimensions [w, h, l]
    [math.pi/2, 0, math.pi/2]  # 5: dimensions [h, l, w]
]


def get_half_ext(original_lwh: list[float], orn_idx: int) -> list[float]:
    if orn_idx == 0:
        return [original_lwh[0]/2, original_lwh[1]/2, original_lwh[2]/2]
    elif orn_idx==1:
        return [original_lwh[0]/2, original_lwh[2]/2, original_lwh[1]/2]
    elif orn_idx==2:
        return [original_lwh[2]/2, original_lwh[1]/2, original_lwh[0]/2]
    elif orn_idx==3:
        return [original_lwh[1]/2, original_lwh[0]/2, original_lwh[2]/2]
    elif orn_idx==4:
        return [original_lwh[1]/2, original_lwh[2]/2, original_lwh[0]/2]
    elif orn_idx==5:
        return [original_lwh[2]/2, original_lwh[0]/2, original_lwh[1]/2]
    else:
        return [original_lwh[0]/2, original_lwh[1]/2, original_lwh[2]/2]


def center_xy(poly, cx, cy):
    return [[x - cx, y - cy] for x, y in poly]


def aff(origin:list[list[float]], rot: list[list[float]], intercept: tuple[float, float, float]):
    aff_transformed = np.dot(np.array(origin), np.array(rot).T) + intercept

    return list(map(tuple, aff_transformed))


def line_intersection_2d(p1, d1, p2, d2):
    """Intersection of the lines p1+t*d1 and p2+s*d2"""
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        raise ValueError("offset failed: parallel lines")
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    t = (dx * d2[1] - dy * d2[0]) / cross
    return [p1[0] + t * d1[0], p1[1] + t * d1[1]]


def offset_convex_polygon_ccw(poly, offset):
    """
    Offset a counter-clockwise convex polygon inwards by `offset` from every edge
    """
    lines = []
    n = len(poly)

    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        ex = b[0] - a[0]
        ey = b[1] - a[1]
        L = math.hypot(ex, ey)
        if L < 1e-12:
            raise ValueError("invalid polygon")

        # inward normal of a CCW polygon
        nx = -ey / L
        ny =  ex / L

        shifted_point = [a[0] + nx * offset, a[1] + ny * offset]
        direction = [ex, ey]
        lines.append((shifted_point, direction))

    inner = []
    for i in range(n):
        p1, d1 = lines[i - 1]
        p2, d2 = lines[i]
        inner.append(line_intersection_2d(p1, d1, p2, d2))
    return inner


def triangulate_fan(indices, reverse=False):
    tris = []
    for i in range(1, len(indices) - 1):
        if reverse:
            tris.append([indices[0], indices[i + 1], indices[i]])
        else:
            tris.append([indices[0], indices[i], indices[i + 1]])
    return tris


def write_open_cut_corner_cup_obj(
    file_path: str,
    width: float = 0.6,          # width of the rectangle
    height: float = 0.6,         # height of the rectangle
    cut_x: float = 0.12,         # amount cut off on the top side
    cut_y: float = 0.12,         # amount cut off on the left side
    depth: float = 0.3,          # depth of the prism (top face to bottom face)
    wall: float = 0.01,          # wall thickness
    bottom: float = 0.01         # bottom thickness
    ) -> tuple[list[list[float]], list[list[float]]]:
    """
    Write an OBJ of a hollow shape with a pentagonal cross-section (upper-left corner cut off), open on top and closed at the bottom
    The cross-section lies in the XY plane and the depth runs along Z
    """

    if wall <= 0 or bottom <= 0:
        raise ValueError("wall, bottom must be > 0")
    if depth <= bottom:
        raise ValueError("depth must be > bottom")
    if cut_x <= 0 or cut_y <= 0:
        raise ValueError("cut_x, cut_y must be > 0")
    if cut_x >= width or cut_y >= height:
        raise ValueError("cut is too large")

    # outer outline (CCW)
    outer2d = [
        [cut_x, 0.0],
        [width, 0.0],
        [width, height],
        [0.0, height],
        [0.0, cut_y],
    ]

    inner2d = offset_convex_polygon_ccw(outer2d, wall)

    # centre the shape (easier to handle when rotating)
    cx = width / 2.0
    cy = height / 2.0


    outer2d = center_xy(outer2d, cx, cy)
    inner2d = center_xy(inner2d, cx, cy)

    z0 = -depth / 2.0            # outer bottom
    z1 =  depth / 2.0            # top edge
    zi = z0 + bottom             # inner bottom

    vertices = []

    # outer bottom
    for x, y in outer2d:
        vertices.append([x, y, z0])

    # outer top
    for x, y in outer2d:
        vertices.append([x, y, z1])

    # inner bottom
    for x, y in inner2d:
        vertices.append([x, y, zi])

    # inner top
    for x, y in inner2d:
        vertices.append([x, y, z1])

    n = 5
    ob = list(range(0, n))         # outer bottom
    ot = list(range(n, 2 * n))     # outer top
    ib = list(range(2 * n, 3 * n)) # inner bottom
    it = list(range(3 * n, 4 * n)) # inner top

    faces = []

    # 1) outer bottom face (underside)
    faces += triangulate_fan(ob, reverse=True)

    # 2) outer side faces
    for i in range(n):
        j = (i + 1) % n
        faces.append([ob[i], ob[j], ot[j]])
        faces.append([ob[i], ot[j], ot[i]])

    # 3) inner bottom face (inside of the cup floor)
    faces += triangulate_fan(ib, reverse=False)

    # 4) inner side faces
    for i in range(n):
        j = (i + 1) % n
        faces.append([ib[i], it[j], ib[j]])
        faces.append([ib[i], it[i], it[j]])

    # 5) rim at the top edge
    # this leaves the top open while keeping a rim as thick as the wall
    for i in range(n):
        j = (i + 1) % n
        faces.append([ot[i], ot[j], it[j]])
        faces.append([ot[i], it[j], it[i]])


    with open(file_path, "w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for tri in faces:
            a, b, c = [idx + 1 for idx in tri]  # OBJ indices start at 1
            f.write(f"f {a} {b} {c}\n")
    m = len(inner2d)
    points = []
    n_vecs = []
    for i in range(m):
        ex = inner2d[(i+1)%m][0]-inner2d[i][0]
        ey = inner2d[(i+1)%m][1]-inner2d[i][1]
        ne = math.hypot(ex, ey)
        n_vec = [ey/ne, -ex/ne, 0]
        points.append(inner2d[i]+[0])
        n_vecs.append(n_vec)
    points.extend([[0,0,z1],[0,0,zi]])
    n_vecs.extend([[0,0,1],[0,0,-1]])

    return points, n_vecs


def write_corner_lid_obj(
    file_path: str,
    width: float = 0.6,
    height: float = 0.3,
    cut_x: float = 0.12,
    cut_y: float = 0.12,
    depth: float = 0.25,
    lid_thickness: float = 0.005
    ):
    """
    A thin lid part (a separate object) that closes the cut corner of the opening
    so that the opening looks rectangular
    """

    outer2d = [
        [cut_x, 0.0],
        [cut_x, height],
        [0.0, height],
        [0.0, cut_y],
    ]

    cx = width / 2.0
    cy = height / 2.0

    outer2d = center_xy(outer2d, cx, cy)

    z_top_open = depth / 2.0
    z0 = z_top_open
    z1 = z_top_open + lid_thickness

    vertices = [
        [outer2d[0][0], outer2d[0][1], z0],  # 0
        [outer2d[1][0], outer2d[1][1], z0],  # 1
        [outer2d[2][0], outer2d[2][1], z0],  # 2
        [outer2d[3][0], outer2d[3][1], z0],  # 3
        [outer2d[0][0], outer2d[0][1], z1],  # 4
        [outer2d[1][0], outer2d[1][1], z1],  # 5
        [outer2d[2][0], outer2d[2][1], z1],  # 6
        [outer2d[3][0], outer2d[3][1], z1],  # 7
    ]

    faces = [
        # bottom face
        [0, 2, 1],
        [0, 3, 2],
        # top face
        [4, 5, 6],
        [4, 6, 7],

        # # side faces
        [0, 1, 4], [4, 1, 5],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [7, 6, 2],
        [3, 0, 4], [4, 7, 3]
    ]

    with open(file_path, "w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for tri in faces:
            a, b, c = [i + 1 for i in tri]
            f.write(f"f {a} {b} {c}\n")
