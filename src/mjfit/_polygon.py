"""Planar polygon kernel shared by the generators.

Polygons are (n, 2) float arrays. Unless a function says otherwise it expects
counterclockwise (CCW) vertex order, the convention every generator uses.
"""
from __future__ import annotations

import math

import mapbox_earcut
import numpy as np

TOL = 1e-12


# ----- basic measures --------------------------------------------------------

def signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly[::-1].copy() if signed_area(poly) < 0 else poly


def is_convex(poly: np.ndarray) -> bool:
    signs = set()
    n = len(poly)
    for i in range(n):
        a, b, c = poly[i], poly[(i + 1) % n], poly[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > TOL:
            signs.add(cross > 0)
    return len(signs) <= 1


def cross2(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def points_equal(a, b, tol: float = 1e-9) -> bool:
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def segments_cross(p1, p2, p3, p4) -> bool:
    """Whether the open segments p1p2 and p3p4 cross in their interiors.

    Touching endpoints and collinear overlaps do not count.
    """
    d1 = cross2(p4[0] - p3[0], p4[1] - p3[1], p1[0] - p3[0], p1[1] - p3[1])
    d2 = cross2(p4[0] - p3[0], p4[1] - p3[1], p2[0] - p3[0], p2[1] - p3[1])
    d3 = cross2(p2[0] - p1[0], p2[1] - p1[1], p3[0] - p1[0], p3[1] - p1[1])
    d4 = cross2(p2[0] - p1[0], p2[1] - p1[1], p4[0] - p1[0], p4[1] - p1[1])
    return ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4))


def edges_cross(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Which edges of polygon p properly cross which edges of polygon q.

    Vectorized `segments_cross` over all edge pairs; shape (len(p), len(q)).
    """
    a, b = p[:, None, :], np.roll(p, -1, axis=0)[:, None, :]
    c, d = q[None, :, :], np.roll(q, -1, axis=0)[None, :, :]

    def cr(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    d1, d2 = cr(d - c, a - c), cr(d - c, b - c)
    d3, d4 = cr(b - a, c - a), cr(b - a, d - a)
    return ((((d1 > 0) & (d2 < 0)) | ((d1 < 0) & (d2 > 0)))
            & (((d3 > 0) & (d4 < 0)) | ((d3 < 0) & (d4 > 0))))


def is_simple(poly: np.ndarray) -> bool:
    """Whether no two edges of the polygon cross (adjacent edges only touch)."""
    return not edges_cross(poly, poly).any()


def points_inside(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Whether each point lies inside the polygon (even-odd rule)."""
    x, y = pts[:, :1], pts[:, 1:2]
    x0, y0 = poly[None, :, 0], poly[None, :, 1]
    x1, y1 = np.roll(poly[:, 0], -1)[None], np.roll(poly[:, 1], -1)[None]
    straddle = (y0 > y) != (y1 > y)
    dy = np.where(y1 != y0, y1 - y0, 1.0)
    crosses = straddle & (x < x0 + (y - y0) * (x1 - x0) / dy)
    return crosses.sum(axis=1) % 2 == 1


def contains(outer: np.ndarray, inner: np.ndarray) -> bool:
    """Whether polygon `inner` lies inside `outer` with no edges crossing."""
    if edges_cross(inner, outer).any():
        return False
    return bool(points_inside(outer, inner).all())


def disjoint(p: np.ndarray, q: np.ndarray) -> bool:
    """Whether two simple polygons share no interior point."""
    if edges_cross(p, q).any():
        return False
    return not points_inside(q, p[:1]).any() and not points_inside(p, q[:1]).any()


def min_rim(outer: np.ndarray, inner: np.ndarray) -> float:
    """Smallest distance from a convex `outer`'s edge lines to `inner`'s vertices."""
    edges = np.roll(outer, -1, axis=0) - outer
    normals = np.column_stack((-edges[:, 1], edges[:, 0]))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return float(np.einsum('ej,evj->ev', normals, inner[None] - outer[:, None]).min())


# ----- validation ------------------------------------------------------------

def outline(points, name: str) -> np.ndarray:
    """Validate a simple polygon and return it CCW, coordinates unchanged."""
    poly = np.asarray(points, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise ValueError(f"{name} needs at least three [x, y] vertices, got shape {poly.shape}.")
    if not np.isfinite(poly).all():
        raise ValueError(f"{name} has non-finite coordinates.")
    gaps = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    if np.any(gaps <= TOL):
        i = int(np.argmax(gaps <= TOL))
        raise ValueError(f"{name} repeats vertex {i} ({poly[i, 0]:g}, {poly[i, 1]:g}).")
    if not is_simple(poly):
        raise ValueError(f"{name} crosses itself; outlines must be simple polygons.")
    if abs(signed_area(poly)) <= TOL:
        raise ValueError(f"{name} has zero area.")
    return ensure_ccw(poly)


def convex_outline(points, name: str) -> np.ndarray:
    """Validate a convex polygon and return it CCW, coordinates unchanged."""
    poly = outline(points, name)
    edges = np.roll(poly, -1, axis=0) - poly
    normals = np.column_stack((-edges[:, 1], edges[:, 0])) / np.linalg.norm(edges, axis=1)[:, None]
    if np.any(np.einsum('ej,evj->ev', normals, poly[None] - poly[:, None]) < -1e-10):
        raise ValueError(f"{name} must be convex.")
    return poly


# ----- offsets ---------------------------------------------------------------

def offset_inward(poly: np.ndarray, distance: float) -> np.ndarray:
    """Move every edge of a CCW polygon `distance` inward along its normal.

    Corners are mitred, so the result has the same vertices and topology and
    every output edge lies exactly `distance` from its input edge. A negative
    distance moves the edges outward.
    """
    edges = np.roll(poly, -1, axis=0) - poly
    dirs = edges / np.linalg.norm(edges, axis=1, keepdims=True)
    inward = np.column_stack([-dirs[:, 1], dirs[:, 0]])
    out = np.zeros_like(poly)
    for i in range(len(poly)):
        d_prev, d_curr = dirs[i - 1], dirs[i]
        a = np.column_stack([d_prev, -d_curr])
        b = distance * (inward[i] - inward[i - 1])
        if abs(np.linalg.det(a)) < TOL:
            out[i] = poly[i] + distance * inward[i - 1]    # collinear edges
        else:
            t, _ = np.linalg.solve(a, b)
            out[i] = poly[i] + distance * inward[i - 1] + t * d_prev
    return out


def offset_outward(poly: np.ndarray, distance: float) -> np.ndarray:
    return offset_inward(poly, -distance)


def offset_is_valid(original: np.ndarray, moved: np.ndarray) -> bool:
    """Whether a mitred offset kept every edge's direction and stayed simple.

    An offset larger than the polygon's local feature size flips edges and
    folds the outline over itself; this detects both.
    """
    e0 = np.roll(original, -1, axis=0) - original
    e1 = np.roll(moved, -1, axis=0) - moved
    if np.any(np.einsum('ij,ij->i', e0, e1) <= 0):
        return False
    return signed_area(moved) > TOL and is_simple(moved)


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Convex hull of 2D points, CCW (Andrew's monotone chain)."""
    pts = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(pts) < 3:
        return np.asarray(pts)

    def chain(seq):
        out: list[tuple[float, float]] = []
        for p in seq:
            while len(out) >= 2 and cross2(out[-1][0] - out[-2][0], out[-1][1] - out[-2][1],
                                           p[0] - out[-1][0], p[1] - out[-1][1]) <= TOL:
                out.pop()
            out.append(p)
        return out
    lower, upper = chain(pts), chain(reversed(pts))
    return np.asarray(lower[:-1] + upper[:-1])


# ----- convex decomposition --------------------------------------------------

def is_star_shaped_from(poly: np.ndarray, point: np.ndarray, tol: float = 1e-9) -> bool:
    """Whether `point` sees every edge of the CCW polygon from inside.

    Then fan triangulation and radial wedges around `point` are valid convex
    decompositions.
    """
    n = len(poly)
    for i in range(n):
        v0, v1 = poly[i], poly[(i + 1) % n]
        d = v1 - v0
        if d[0] * (point[1] - v0[1]) - d[1] * (point[0] - v0[0]) <= tol:
            return False
    return True


def point_in_triangle(p, a, b, c, tol: float = TOL) -> bool:
    def s(q1, q2, q3):
        return (q1[0] - q3[0]) * (q2[1] - q3[1]) - (q2[0] - q3[0]) * (q1[1] - q3[1])
    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    has_neg = d1 < -tol or d2 < -tol or d3 < -tol
    has_pos = d1 > tol or d2 > tol or d3 > tol
    return not (has_neg and has_pos)


def ear_clip(poly: np.ndarray) -> list[np.ndarray]:
    """Triangulate a simple polygon by ear clipping (degenerate ears dropped)."""
    if signed_area(poly) < 0:
        poly = poly[::-1]
    verts = [tuple(v) for v in poly]
    triangles = []
    budget = len(verts) ** 2 + 20
    while len(verts) > 3:
        budget -= 1
        if budget <= 0:
            raise RuntimeError("Ear clipping did not terminate.")
        n = len(verts)
        for i in range(n):
            a, b, c = verts[(i - 1) % n], verts[i], verts[(i + 1) % n]
            if cross2(b[0] - a[0], b[1] - a[1], c[0] - b[0], c[1] - b[1]) <= TOL:
                continue                     # reflex or flat corner
            blocked = False
            for j in range(n):
                if j in ((i - 1) % n, i, (i + 1) % n):
                    continue
                vj = verts[j]
                if points_equal(vj, a) or points_equal(vj, b) or points_equal(vj, c):
                    continue
                if point_in_triangle(vj, a, b, c):
                    blocked = True
                    break
            if not blocked:
                triangles.append(np.array([a, b, c]))
                verts.pop(i)
                break
        else:
            raise RuntimeError("No ear found; the polygon may cross itself.")
    triangles.append(np.array(verts))
    return [t for t in triangles if abs(signed_area(t)) > 1e-14]


def triangulate(outer: np.ndarray, holes: list[np.ndarray] = ()) -> list[np.ndarray]:
    """Triangulate a CCW polygon minus CCW holes (degenerate triangles dropped)."""
    if not len(holes):
        return ear_clip(outer)
    rings = [np.asarray(outer, dtype=np.float64)]
    rings += [np.asarray(h, dtype=np.float64)[::-1] for h in holes]   # holes CW
    vertices = np.concatenate(rings, axis=0)
    ring_end = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    idx = mapbox_earcut.triangulate_float64(vertices, ring_end)
    triangles = [vertices[idx[i:i + 3]] for i in range(0, len(idx), 3)]
    return [ensure_ccw(t) for t in triangles if abs(2 * signed_area(t)) > TOL]


def fan(poly: np.ndarray) -> list[np.ndarray]:
    """Triangles from the vertex mean to every edge (convex for star shapes)."""
    c = poly.mean(axis=0)
    n = len(poly)
    return [np.array([c, poly[i], poly[(i + 1) % n]]) for i in range(n)]


def convex_pieces(poly: np.ndarray) -> list[np.ndarray]:
    """Convex pieces whose union is the polygon: itself, a fan, or ear clips."""
    if is_convex(poly):
        return [poly]
    if is_star_shaped_from(poly, poly.mean(axis=0)):
        return fan(poly)
    return ear_clip(poly)


def _ray_to_square(origin: np.ndarray, direction: np.ndarray, half: float) -> np.ndarray:
    """Where the ray origin + t * direction (t > 0) leaves the square [-half, half]^2."""
    t_best, best = math.inf, None
    for axis in range(2):
        d = direction[axis]
        if abs(d) < TOL:
            continue
        for bound in (-half, half):
            t = (bound - origin[axis]) / d
            if t <= 1e-9:
                continue
            pt = origin + t * direction
            if -half - 1e-9 <= pt[1 - axis] <= half + 1e-9 and t < t_best:
                t_best, best = t, pt
    if best is None:
        raise ValueError("A ray from the hole center misses the plate.")
    return best


def radial_wedges(hole: np.ndarray, half: float) -> list[np.ndarray]:
    """Square plate [-half, half]^2 minus a hole, cut into one wedge per hole edge.

    The hole must be star-shaped from its vertex mean: each wedge is bounded
    by one hole edge, the rays from the mean through its two vertices, and
    the plate boundary (including any plate corners between the rays).
    """
    center = hole.mean(axis=0)
    corners = np.array([[half, half], [-half, half], [-half, -half], [half, -half]])

    def angle(p):
        return math.atan2(p[1] - center[1], p[0] - center[0])

    wedges = []
    n = len(hole)
    for i in range(n):
        v0, v1 = hole[i], hole[(i + 1) % n]
        a0 = angle(v0)
        span = (angle(v1) - a0) % (2 * math.pi)
        between = []
        for c in corners:
            off = (angle(c) - a0) % (2 * math.pi)
            if 1e-9 < off < span - 1e-9:
                between.append((off, c))
        # Walk the wedge CCW: out along the ray through v0, along the plate
        # boundary to the ray through v1, then in and back along the hole edge.
        verts = [_ray_to_square(center, v0 - center, half)]
        verts += [c for _, c in sorted(between, key=lambda e: e[0])]
        verts += [_ray_to_square(center, v1 - center, half), v1, v0]
        wedges.append(np.asarray(verts))
    return wedges
