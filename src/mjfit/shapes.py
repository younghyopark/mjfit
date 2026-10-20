"""Planar outlines for pegs, holes, islands, handles and plates.

Every function returns a counterclockwise ``(n, 2)`` array of vertices in
meters. Angles are in degrees. Curves become polygons that *circumscribe*
them: the flats touch the curve and the corners stick out by at most
``max_error``. A hole is therefore never smaller than drawn (and an island
never thinner), and since the matching peg is an exact offset of the hole,
the clearance is exact along every flat and only grows at the corners.
Against the curves themselves, though, each part is off by up to
``max_error``, so keep it well below the clearance.

The exception is :func:`bezier`, whose vertices lie on the curve (the chords
cut inside it by at most ``max_error``): there the polygon is the shape.

In a config file an outline is either a list of vertices or a mapping that
names one of these functions, e.g. ``{type: circle, radius: 0.008}``.
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np

from . import _polygon as pg

__all__ = ['circle', 'rect', 'stadium', 'trapezoid', 'ngon', 'star', 'polygon',
           'bezier', 'csv', 'preset', 'from_spec', 'PRESETS']

MAX_ERROR = 1e-4
"""Default bound on how far a polygon may deviate from the curve it replaces (m)."""


def _place(poly: np.ndarray, center, rotate: float) -> np.ndarray:
    poly = np.asarray(poly, dtype=float)
    if rotate:
        a = math.radians(rotate)
        c, s = math.cos(a), math.sin(a)
        poly = poly @ np.array([[c, s], [-s, c]])
    return pg.ensure_ccw(poly + np.asarray(center, dtype=float))


def _sides(radius: float, max_error: float, minimum: int = 6) -> int:
    """Fewest sides whose circumscribed polygon stays within max_error of the circle."""
    if not max_error > 0:
        raise ValueError(f"max_error must be positive, got {max_error!r}.")
    half_angle = math.acos(radius / (radius + max_error))
    return max(minimum, math.ceil(math.pi / half_angle))


def _positive(**values: float) -> None:
    for name, value in values.items():
        if not (isinstance(value, (int, float, np.floating, np.integer)) and math.isfinite(value)
                and value > 0):
            raise ValueError(f"{name} must be a positive number, got {value!r}.")


def circle(radius: float, *, n: int | None = None, max_error: float = MAX_ERROR,
           center=(0.0, 0.0), rotate: float = 0.0) -> np.ndarray:
    """Regular ``n``-gon circumscribing a circle (first vertex on +x).

    Without ``n``, uses the fewest sides whose corners stick out by at most
    ``max_error``.
    """
    _positive(radius=radius)
    n = _sides(radius, max_error) if n is None else int(n)
    if n < 3:
        raise ValueError(f"A circle needs n >= 3 sides, got {n}.")
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    r = radius / math.cos(math.pi / n)
    return _place(np.column_stack([r * np.cos(a), r * np.sin(a)]), center, rotate)


def rect(width: float, height: float, *, center=(0.0, 0.0), rotate: float = 0.0) -> np.ndarray:
    """Axis-aligned rectangle centered on the origin."""
    _positive(width=width, height=height)
    w, h = width / 2, height / 2
    return _place([[-w, -h], [w, -h], [w, h], [-w, h]], center, rotate)


def stadium(width: float, height: float, *, n: int | None = None,
            max_error: float = MAX_ERROR, center=(0.0, 0.0), rotate: float = 0.0) -> np.ndarray:
    """Rectangle with semicircular ends along its longer side.

    The two ends together are a circumscribed ``n``-gon (``n`` even) split
    at its flats, so the straight sides are exact.
    """
    _positive(width=width, height=height)
    long, short = max(width, height), min(width, height)
    r = short / 2
    if n is None:
        n = _sides(r, max_error)
    n = int(n) + int(n) % 2
    if n < 4:
        raise ValueError(f"A stadium needs n >= 4, got {n}.")
    step = 2 * np.pi / n
    a = -np.pi / 2 + (np.arange(n) + 0.5) * step         # flats at +-90 degrees
    pts = np.column_stack([np.cos(a), np.sin(a)]) * (r / math.cos(step / 2))
    pts[:, 0] += np.where(pts[:, 0] > 0, 1.0, -1.0) * (long - short) / 2
    if height > width:
        pts = pts[:, ::-1] * [-1.0, 1.0]
    return _place(pts, center, rotate)


def trapezoid(bottom_width: float, top_width: float, height: float, *,
              center=(0.0, 0.0), rotate: float = 0.0) -> np.ndarray:
    """Isosceles trapezoid with its parallel sides along x."""
    _positive(bottom_width=bottom_width, top_width=top_width, height=height)
    b, t, h = bottom_width / 2, top_width / 2, height / 2
    return _place([[-b, -h], [b, -h], [t, h], [-t, h]], center, rotate)


def ngon(n: int, radius: float, *, center=(0.0, 0.0), rotate: float = 0.0) -> np.ndarray:
    """Regular polygon with circumradius ``radius`` (first vertex on +y)."""
    _positive(radius=radius)
    if int(n) < 3:
        raise ValueError(f"A polygon needs n >= 3 sides, got {n}.")
    a = np.pi / 2 + np.linspace(0.0, 2 * np.pi, int(n), endpoint=False)
    return _place(np.column_stack([radius * np.cos(a), radius * np.sin(a)]), center, rotate)


def star(points: int, outer_radius: float, inner_radius: float, *,
         center=(0.0, 0.0), rotate: float = 0.0) -> np.ndarray:
    """Star with ``points`` tips, alternating between the two radii (a tip on +y)."""
    _positive(outer_radius=outer_radius, inner_radius=inner_radius)
    if int(points) < 3 or inner_radius >= outer_radius:
        raise ValueError("A star needs points >= 3 and inner_radius < outer_radius.")
    a = np.pi / 2 + np.linspace(0.0, 2 * np.pi, 2 * int(points), endpoint=False)
    r = np.where(np.arange(2 * int(points)) % 2 == 0, outer_radius, inner_radius)
    return _place(np.column_stack([r * np.cos(a), r * np.sin(a)]), center, rotate)


def polygon(vertices, *, center=(0.0, 0.0), rotate: float = 0.0) -> np.ndarray:
    """Any simple polygon, given by its vertices in either order."""
    return _place(pg.outline(vertices, 'Polygon'), center, rotate)


def bezier(vertices, rounding=0.5, *, max_error: float = MAX_ERROR,
           center=(0.0, 0.0), rotate: float = 0.0) -> np.ndarray:
    """Convex polygon whose corners are rounded by quadratic Bezier curves.

    ``rounding`` is one fraction or one per vertex, in [0, 0.5]: 0 keeps the
    corner sharp, 0.5 starts the curve at the middle of both adjacent edges.
    The curves are sampled so that no chord cuts more than ``max_error``
    inside them.
    """
    original = np.asarray(vertices, dtype=float)
    control = pg.convex_outline(original, 'Bezier control polygon')
    alphas = np.asarray(rounding, dtype=float)
    if alphas.ndim == 0:
        alphas = np.full(len(control), float(alphas))
    if (alphas.shape != (len(control),) or not np.isfinite(alphas).all()
            or np.any((alphas < 0) | (alphas > 0.5))):
        raise ValueError("rounding needs fractions in [0, 0.5], one or one per vertex.")
    if pg.signed_area(original) < 0:
        alphas = alphas[::-1]                 # follow the vertices into CCW order
    if not (math.isfinite(max_error) and max_error > 0):
        raise ValueError("max_error must be finite and positive.")
    points: list[np.ndarray] = []
    for i, (vertex, alpha) in enumerate(zip(control, alphas)):
        if alpha == 0:
            samples = [vertex]
        else:
            entry = vertex + alpha * (control[i - 1] - vertex)
            exit_ = vertex + alpha * (control[(i + 1) % len(control)] - vertex)
            curvature = np.linalg.norm(entry - 2 * vertex + exit_)
            segments = max(1, math.ceil(math.sqrt(curvature / (4 * max_error))))
            samples = [(1 - t) ** 2 * entry + 2 * t * (1 - t) * vertex + t ** 2 * exit_
                       for t in np.linspace(0.0, 1.0, segments + 1)]
        for p in samples:
            if not points or np.linalg.norm(p - points[-1]) > pg.TOL:
                points.append(p)
    if np.linalg.norm(points[0] - points[-1]) <= pg.TOL:
        points.pop()
    poly = pg.convex_outline(points, 'Rounded polygon')
    return _place(poly, center, rotate) if (rotate or np.any(center)) else poly


def csv(path: str | Path, *, center=(0.0, 0.0), rotate: float = 0.0) -> np.ndarray:
    """Polygon read from a CSV file with one ``x,y`` vertex per line."""
    return polygon(np.loadtxt(Path(path), delimiter=','), center=center, rotate=rotate)


PRESETS = ('circle', 'triangle', 'square', 'pentagon', 'hexagon', 'octagon', 'star', 'slot')


def preset(name: str, size: float) -> np.ndarray:
    """A named hole for quick experiments; ``size`` is its circumradius.

    ``star`` has five points with an inner radius of 0.45 ``size``; ``slot``
    is a stadium ``2 size`` long and ``0.7 size`` wide.
    """
    sides = {'triangle': 3, 'square': 4, 'pentagon': 5, 'hexagon': 6, 'octagon': 8}
    if name == 'circle':
        return circle(size)
    if name in sides:
        return ngon(sides[name], size)
    if name == 'star':
        return star(5, size, 0.45 * size)
    if name == 'slot':
        return stadium(2 * size, 0.7 * size)
    raise ValueError(f"Unknown preset {name!r}; choose from {', '.join(PRESETS)}.")


_TYPES = {f.__name__: f for f in (circle, rect, stadium, trapezoid, ngon, star, polygon,
                                  bezier, csv)}


def from_spec(spec, base_dir: str | Path | None = None, name: str = 'outline') -> np.ndarray:
    """Resolve an outline: a vertex list, or a mapping naming a shape function.

    Relative CSV paths are read from ``base_dir`` (a config file's folder).
    """
    if not isinstance(spec, dict):
        return pg.outline(spec, name.capitalize())
    spec = dict(spec)
    kind = spec.pop('type', None)
    if kind not in _TYPES:
        raise ValueError(f"{name}: unknown shape type {kind!r}; choose from "
                         f"{', '.join(_TYPES)}.")
    fn = _TYPES[kind]
    params = inspect.signature(fn).parameters
    unknown = sorted(set(spec) - set(params))
    if unknown:
        raise ValueError(f"{name}: {kind} takes {', '.join(params)}; "
                         f"unknown key(s) {', '.join(unknown)}.")
    if kind == 'csv' and base_dir is not None and not Path(spec['path']).is_absolute():
        spec['path'] = Path(base_dir) / spec['path']
    try:
        return fn(**spec)
    except TypeError as e:
        raise ValueError(f"{name}: {e}") from None
