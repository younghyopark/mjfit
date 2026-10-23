"""Spur gears, internal ring gears, and meshing gear pairs."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from . import _mjcf as mj
from . import _polygon as pg

RGBA = ('0.55 0.58 0.62 1', '0.78 0.58 0.26 1')
HUB_RGBA = '0.42 0.44 0.48 1'
AXLE_RGBA = '0.18 0.20 0.23 1'
FLOOR_GAP = 0.001
FLANK_SAMPLES, ROOT_SAMPLES, TIP_SAMPLES = 4, 2, 2


def gear(
    *,
    teeth=(18, 24),
    module: float = 0.003,
    face_width: float = 0.008,
    pressure_angle: float = 20.0,
    addendum: float = 1.0,
    dedendum: float = 1.25,
    tooth_width: float = 0.48,
    center_distance: float | None = None,
    internal: bool = False,
    rim: float = 2.5,
    density: float = 7800.0,
    axles: bool = True,
    motor: bool = True,
    motor_gear: float = 0.02,
    damping: float = 0.001,
    armature: float = 1e-5,
    friction=0.8,
    condim: int = 3,
    timestep: float = 0.001,
    standalone: bool = True,
    floor: bool = True,
    mesh_dir: str | Path | None = None,
    scale: float = 1.0,
) -> str:
    """A spur gear, a ring gear, or a meshing pair, as an MJCF string.

    Each gear turns on a hinge about +z. An external gear's collision shape is
    one convex prism per tooth plus one for the hub inside the root circle. A
    ring gear (``internal=True``) is convex wedges from each inner-edge out to
    the rim. A separate visual mesh renders the outline (the collision prisms
    are in geom group 3).

    Hinged gears give MuJoCo's default Newton solver almost no contact
    regularization (their mass sits on the axis), so standalone files use the
    PGS solver, which meshes them at the exact ratio. An internal pair turns
    both gears the same way.

    Args:
        teeth: Tooth count of one gear, or two counts for a meshing pair. With
            ``internal=True`` and two counts, the first is the pinion and the
            second is the ring (it must have more teeth).
        module: Pitch diameter per tooth, in meters; shared by both gears.
        face_width: Thickness of the gears.
        pressure_angle: In degrees.
        addendum: Tooth height from the pitch circle, in modules. On a ring
            this is inward, toward the axis.
        dedendum: Depth from the pitch circle, in modules. On a ring this is
            outward, into the rim.
        tooth_width: Tooth thickness at the pitch circle, as a fraction of the
            circular pitch; below 0.5 leaves backlash.
        center_distance: Distance between the axes. Default: the sum of the
            pitch radii, or their difference for an internal pair.
        internal: Build a ring gear with inward teeth. A pair is then a pinion
            inside the ring.
        rim: Radial thickness of a ring gear outside the tooth roots, in
            modules. Ignored for external gears.
        density: kg/m^3, or one per gear.
        axles: Draw the axles (visual only).
        motor: Add a motor on the first gear's hinge (control in [-1, 1]).
        motor_gear: Torque per unit of motor control, N m.
        damping: Hinge damping.
        armature: Hinge armature (rotor inertia).
        friction: Sliding friction, or the (sliding, torsional, rolling) triple.
        condim: MuJoCo contact dimensionality (1, 3, 4 or 6).
        timestep: Simulation time step in seconds (standalone files).
        standalone: Add a floor, a light and solver options so the file runs on
            its own. False emits only the parts.
        floor: Add a ground plane (standalone files).
        mesh_dir: Write meshes as OBJ files into this folder, to be saved next
            to the XML file, instead of inline vertex lists.
        scale: Multiplies every length.

    Returns:
        The MJCF model as a string, with bodies ``gear_a`` (and ``gear_b``) on
        hinges ``gear_a_hinge`` (and ``gear_b_hinge``).
    """
    counts = [int(t) for t in np.atleast_1d(teeth)]
    if len(counts) not in (1, 2) or any(t != t_in for t, t_in in zip(counts, np.atleast_1d(teeth))):
        raise ValueError(f"teeth must be one or two integers, got {teeth!r}.")
    if rim <= 0:
        raise ValueError("rim must be positive.")
    densities = [float(d) for d in np.broadcast_to(np.atleast_1d(density), (len(counts),))]
    if face_width <= 0 or scale <= 0:
        raise ValueError("face_width and scale must be positive.")
    m, width = module * scale, face_width * scale
    ring = [internal] * len(counts) if len(counts) == 1 else [False, internal]
    if internal and len(counts) == 2 and counts[1] <= counts[0]:
        raise ValueError("The ring (second tooth count) must have more teeth than the pinion.")
    dims = [_dimensions(t, m, math.radians(pressure_angle), addendum, dedendum, tooth_width,
                        internal=is_ring, rim=rim)
            for t, is_ring in zip(counts, ring)]
    pair = len(dims) == 2
    if center_distance is None:
        if not pair:
            distance = 0.0
        elif internal:
            distance = dims[1]['pitch_radius'] - dims[0]['pitch_radius']
        else:
            distance = dims[0]['pitch_radius'] + dims[1]['pitch_radius']
    else:
        distance = center_distance * scale
        if not distance > 0:
            raise ValueError("center_distance must be positive.")
    if pair and internal:
        if distance + dims[0]['outer_radius'] > dims[1]['root_radius']:
            raise ValueError("center_distance is too large: the pinion would reach the ring root.")
    elif pair and distance <= max(dims[0]['outer_radius'] + dims[1]['root_radius'],
                                  dims[1]['outer_radius'] + dims[0]['root_radius']):
        raise ValueError("center_distance is too small: teeth would reach the other gear's root.")
    xs = [-distance / 2, distance / 2] if pair else [0.0]
    # External: a gap of the second gear on the line of centers, where the
    # first gear's tooth at phase zero points. Internal: the pitch circles
    # touch on the far side of the pinion, so the pinion tooth points -x and
    # a ring space faces the same way.
    if not pair:
        phases = [0.0]
    elif internal:
        phases = [math.pi, math.pi]
    else:
        phases = [0.0, math.pi + math.pi / counts[1]]

    model = mj.Model('gears')
    defaults = model.contact_defaults(friction=friction, condim=condim, visual_classes=True)
    ET.SubElement(defaults, 'joint', damping=mj.num(damping), armature=mj.num(armature))
    meshes = mj.Meshes(model, mesh_dir)
    z = width * 1.25 / 2 + FLOOR_GAP                 # the hub is 1.25 face widths
    if standalone:
        reach = max(abs(x) + d['extent'] for x, d in zip(xs, dims))
        model.standalone(timestep=timestep, floor=floor, floor_size=max(0.4, 4 * reach),
                         light_pos=(0.2, -0.35, 0.35), solver='PGS')
    wb = model.section('worldbody')
    for k, (d, x, phase, rho) in enumerate(zip(dims, xs, phases, densities)):
        name = f'gear_{"ab"[k]}'
        if axles:
            height = z + width * 1.25 / 2 + 0.004 * scale
            ET.SubElement(wb, 'geom', {'name': f'{name}_axle', 'type': 'cylinder',
                                       'class': mj.VISUAL, 'rgba': AXLE_RGBA,
                                       'size': mj.vec((0.45 * m, height / 2)),
                                       'pos': mj.vec((x, 0, height / 2))})
        body = ET.SubElement(wb, 'body', name=name, childclass=mj.CLASS, pos=mj.vec((x, 0, z)))
        ET.SubElement(body, 'joint', name=f'{name}_hinge', type='hinge', axis='0 0 1')
        if d['internal']:
            hole = _internal_hole_outline(d, phase)
            rim_poly = _rim_polygon(d)
            visual = meshes.solid(f'{name}_visual', [(rim_poly, [hole], -width / 2, width / 2)])
            pieces = _internal_pieces(d, phase)
        else:
            outline = _outline(d, phase)
            visual = meshes.solid(f'{name}_visual', [(outline, [], -width / 2, width / 2)])
            pieces = _pieces(outline, d['root_radius'])
            ET.SubElement(body, 'geom', {'name': f'{name}_hub', 'type': 'cylinder',
                                         'class': mj.VISUAL, 'rgba': HUB_RGBA,
                                         'size': mj.vec((0.35 * d['root_radius'],
                                                         width * 1.25 / 2))})
        ET.SubElement(body, 'geom', {'name': f'{name}_visual', 'type': 'mesh', 'mesh': visual,
                                     'class': mj.VISUAL, 'rgba': RGBA[k]})
        for i, piece in enumerate(pieces):
            mesh = meshes.prism(f'{name}_col_{i}', piece, -width / 2, width / 2)
            ET.SubElement(body, 'geom', {'name': mesh, 'type': 'mesh', 'mesh': mesh,
                                         'class': mj.COLLISION, 'density': mj.num(rho)})
    if motor:
        ET.SubElement(model.section('actuator'), 'motor', name='drive_gear_a',
                      joint='gear_a_hinge', gear=mj.num(motor_gear), ctrlrange='-1 1')
    return model.to_xml()


def _dimensions(teeth: int, module: float, pressure_angle: float, addendum: float,
                dedendum: float, tooth_width: float, *, internal: bool = False,
                rim: float = 2.5) -> dict:
    if teeth < 3:
        raise ValueError("A gear needs at least 3 teeth.")
    if module <= 0:
        raise ValueError("module must be positive.")
    if not 0 < pressure_angle < math.radians(45):
        raise ValueError("pressure_angle must be between 0 and 45 degrees.")
    if addendum <= 0 or dedendum <= 0:
        raise ValueError("addendum and dedendum must be positive.")
    if not 0.1 < tooth_width < 0.8:
        raise ValueError("tooth_width must be between 0.1 and 0.8.")
    pitch_radius = module * teeth / 2
    if internal:
        tip_radius = pitch_radius - addendum * module
        root_radius = pitch_radius + dedendum * module
        if tip_radius <= module * 0.25:
            raise ValueError(f"The tip circle of a {teeth}-tooth ring is too small; "
                             "use more teeth or a smaller addendum.")
        rim_radius = root_radius + rim * module
        extent, outer_radius = rim_radius, tip_radius
    else:
        tip_radius = pitch_radius + addendum * module
        root_radius = pitch_radius - dedendum * module
        if root_radius <= module * 0.25:
            raise ValueError(f"The root circle of a {teeth}-tooth gear is too small; "
                             "use more teeth or a smaller dedendum.")
        rim_radius = root_radius
        extent, outer_radius = tip_radius, tip_radius
    return {
        'teeth': teeth,
        'internal': internal,
        'pressure_angle': pressure_angle,
        'pitch_radius': pitch_radius,
        'base_radius': pitch_radius * math.cos(pressure_angle),
        'root_radius': root_radius,
        'tip_radius': tip_radius,
        'outer_radius': outer_radius,
        'rim_radius': rim_radius,
        'extent': extent,
        'half_tooth_angle_pitch': math.pi / teeth * tooth_width,
    }


def _involute(a: float) -> float:
    return math.tan(a) - a


def _half_angle(radius: float, d: dict) -> float:
    """Half the angular thickness of a tooth at a radius (involute relation).

    External teeth get thinner toward the tip (larger radius). Internal teeth
    get thinner toward the tip too, which is the smaller radius, so the
    involute term flips sign.
    """
    base = d['base_radius']
    inv_pitch = _involute(d['pressure_angle'])
    sign = -1.0 if d.get('internal') else 1.0
    if radius <= base:
        return d['half_tooth_angle_pitch'] + sign * inv_pitch
    alpha = math.acos(max(-1.0, min(1.0, base / radius)))
    return d['half_tooth_angle_pitch'] + sign * (inv_pitch - _involute(alpha))


def _outline(d: dict, phase: float) -> np.ndarray:
    """CCW gear outline: involute flanks above the base circle, radial below
    it, arcs at the root and the tip."""
    teeth = d['teeth']
    root, base, outer = d['root_radius'], d['base_radius'], d['outer_radius']
    pitch = 2 * math.pi / teeth
    start = max(root, base)
    theta_root = _half_angle(start, d)
    if theta_root >= pitch * 0.48:
        raise ValueError("The tooth base fills almost the whole pitch; reduce tooth_width, "
                         "pressure_angle or dedendum.")
    pts: list[tuple[float, float]] = []

    def add(radius: float, angle: float) -> None:
        p = (radius * math.cos(angle), radius * math.sin(angle))
        if not pts or abs(p[0] - pts[-1][0]) > 1e-12 or abs(p[1] - pts[-1][1]) > 1e-12:
            pts.append(p)

    for tooth in range(teeth):
        center = phase + tooth * pitch
        prev_right, left = center - pitch + theta_root, center - theta_root
        for j in range(1, ROOT_SAMPLES + 1):
            add(root, prev_right + (left - prev_right) * j / ROOT_SAMPLES)
        if root < base:
            add(base, left)
        for j in range(FLANK_SAMPLES + 1):
            r = start + (outer - start) * j / FLANK_SAMPLES
            add(r, center - _half_angle(r, d))
        theta_tip = _half_angle(outer, d)
        for j in range(1, TIP_SAMPLES + 1):
            add(outer, center - theta_tip + 2 * theta_tip * j / TIP_SAMPLES)
        for j in range(1, FLANK_SAMPLES + 1):
            r = outer - (outer - start) * j / FLANK_SAMPLES
            add(r, center + _half_angle(r, d))
        if root < base:
            add(root, center + theta_root)
    return np.asarray(pts)


def _pieces(outline: np.ndarray, root_radius: float) -> list[np.ndarray]:
    """Convex pieces: the hub polygon through every root-circle point, and one
    piece per tooth between consecutive root points. Falls back to a fan
    around the center if a tooth is not convex."""
    on_root = np.abs(np.linalg.norm(outline, axis=1) - root_radius) < 1e-9 * max(root_radius, 1e-3)
    roots = np.flatnonzero(on_root)
    hub = outline[roots]
    teeth = []
    for a, b in zip(roots, np.roll(roots, -1)):
        run = outline[a:b + 1] if b > a else np.vstack([outline[a:], outline[:b + 1]])
        if len(run) > 2:
            teeth.append(run)
    if all(pg.is_convex(t) for t in teeth) and pg.is_convex(hub):
        return [hub] + teeth
    center = np.zeros(2)
    n = len(outline)
    return [np.array([center, outline[i], outline[(i + 1) % n]]) for i in range(n)
            if abs(pg.signed_area(np.array([center, outline[i], outline[(i + 1) % n]]))) > 1e-14]


def _rim_polygon(d: dict) -> np.ndarray:
    """CCW outer rim of a ring gear."""
    n = max(64, int(d['teeth']) * 4)
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    r = d['rim_radius']
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def _internal_hole_outline(d: dict, phase: float) -> np.ndarray:
    """CCW outline of the empty space inside a ring gear.

    The hole is an external gear of the same pitch radius whose teeth are the
    ring's spaces: they reach the ring root, their valleys are the ring tips,
    and their thickness is the complement of ``tooth_width``. The flanks are
    then the same involute as an external tooth on this base circle.
    """
    pitch = math.pi / d['teeth']
    hole = dict(d)
    hole['internal'] = False
    hole['root_radius'] = d['tip_radius']
    hole['outer_radius'] = d['root_radius']
    hole['half_tooth_angle_pitch'] = pitch - d['half_tooth_angle_pitch']
    return _outline(hole, phase)


def _internal_pieces(d: dict, phase: float) -> list[np.ndarray]:
    """Convex wedges of the ring: each hole edge extruded out to the rim.

    The hole is star-shaped from the origin, so the ray through each vertex
    stays in the material. Falls back to triangulating the rim minus the hole
    if a wedge is not convex.
    """
    hole = _internal_hole_outline(d, phase)
    rim = d['rim_radius']
    pieces = []
    n = len(hole)
    for i in range(n):
        a, b = hole[i], hole[(i + 1) % n]
        ra, rb = np.linalg.norm(a), np.linalg.norm(b)
        if ra < 1e-18 or rb < 1e-18:
            continue
        poly = pg.ensure_ccw(np.array([a, b, b * (rim / rb), a * (rim / ra)]))
        if abs(pg.signed_area(poly)) > 1e-14:
            pieces.append(poly)
    if all(pg.is_convex(p) for p in pieces):
        return pieces
    return pg.triangulate(_rim_polygon(d), [hole])
