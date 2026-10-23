"""Spur gears, racks, and meshing gear pairs."""
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
    rack_teeth: int | None = None,
    rack_backing: float = 2.0,
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
    """A spur gear, a rack and pinion, or a meshing pair, as an MJCF string.

    Each gear turns on a hinge about +z. Its collision shape is one convex
    prism per tooth plus one for the hub inside the root circle; a separate
    visual mesh renders the outline (the collision prisms are in geom group 3).
    A rack (``rack_teeth``) slides along +x; its teeth are convex trapezoids
    on a backing bar, and it meshes with the pinion at the pitch line.

    Hinged gears give MuJoCo's default Newton solver almost no contact
    regularization (their mass sits on the axis), so standalone files use the
    PGS solver, which meshes them at the exact ratio.

    Args:
        teeth: Tooth count of one gear, or two counts for a meshing pair. With
            ``rack_teeth``, this is the pinion's count only.
        module: Pitch diameter per tooth, in meters; shared by both gears.
        face_width: Thickness of the gears.
        pressure_angle: In degrees.
        addendum: Tooth height above the pitch circle, in modules.
        dedendum: Depth below the pitch circle, in modules.
        tooth_width: Tooth thickness at the pitch circle, as a fraction of the
            circular pitch; below 0.5 leaves backlash.
        center_distance: Distance between the axes. Default: the sum of the
            pitch radii. With a rack, the height of the pinion centre above
            the rack pitch line (default: the pinion's pitch radius).
        rack_teeth: If set, build a linear rack with this many teeth and a
            pinion of ``teeth`` (a single count). The rack body is ``rack``
            on slide joint ``rack_slide``.
        rack_backing: Thickness of the rack bar below the tooth roots, in
            modules.
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
        The MJCF model as a string, with bodies ``gear_a`` (and ``gear_b`` or
        ``rack``) on hinges ``gear_a_hinge`` (and ``gear_b_hinge``) or slide
        ``rack_slide``.
    """
    counts = [int(t) for t in np.atleast_1d(teeth)]
    if len(counts) not in (1, 2) or any(t != t_in for t, t_in in zip(counts, np.atleast_1d(teeth))):
        raise ValueError(f"teeth must be one or two integers, got {teeth!r}.")
    n_rack = None
    if rack_teeth is not None:
        n_rack = int(rack_teeth)
        if n_rack != rack_teeth or n_rack < 3:
            raise ValueError("rack_teeth must be an integer >= 3.")
        if len(counts) != 1:
            raise ValueError("rack_teeth needs one pinion tooth count, not a pair.")
        if rack_backing <= 0:
            raise ValueError("rack_backing must be positive.")
    n_parts = 2 if n_rack is not None else len(counts)
    densities = [float(d) for d in np.broadcast_to(np.atleast_1d(density), (n_parts,))]
    if face_width <= 0 or scale <= 0:
        raise ValueError("face_width and scale must be positive.")
    m, width = module * scale, face_width * scale
    pa = math.radians(pressure_angle)
    dims = [_dimensions(t, m, pa, addendum, dedendum, tooth_width) for t in counts]
    pair = len(dims) == 2
    backing = rack_backing * m
    if center_distance is None:
        distance = dims[0]['pitch_radius'] + dims[1]['pitch_radius'] if pair else 0.0
    else:
        distance = center_distance * scale
        if not distance > 0:
            raise ValueError("center_distance must be positive.")
    if pair and distance <= max(dims[0]['outer_radius'] + dims[1]['root_radius'],
                                dims[1]['outer_radius'] + dims[0]['root_radius']):
        raise ValueError("center_distance is too small: teeth would reach the other gear's root.")
    xs = [-distance / 2, distance / 2] if pair else [0.0]
    # Put a gap of the second gear on the line of centers, where the first
    # gear's tooth at phase zero points.
    phases = [0.0, math.pi + math.pi / counts[1]] if pair else [0.0]

    model = mj.Model('gears')
    defaults = model.contact_defaults(friction=friction, condim=condim, visual_classes=True)
    ET.SubElement(defaults, 'joint', damping=mj.num(damping), armature=mj.num(armature))
    meshes = mj.Meshes(model, mesh_dir)
    z = width * 1.25 / 2 + FLOOR_GAP                 # the hub is 1.25 face widths
    if n_rack is not None:
        pinion = dims[0]
        pitch_line = pinion['pitch_radius'] if center_distance is None else center_distance * scale
        if pitch_line - pinion['outer_radius'] < -dedendum * m:
            raise ValueError("center_distance is too small: the pinion would reach the rack root.")
        pitch = math.pi * m
        half_len = 0.5 * n_rack * pitch
        if standalone:
            reach = max(half_len, pitch_line + pinion['outer_radius']) + 0.02 * scale
            model.standalone(timestep=timestep, floor=floor, floor_size=max(0.4, 4 * reach),
                             light_pos=(0.2, -0.35, 0.35), solver='PGS')
        wb = model.section('worldbody')
        # Tooth at -y, plus a small rotation so a pinion tooth sits in a rack space.
        _emit_pinion(wb, meshes, pinion, name='gear_a', x=0.0, y=pitch_line, z=z,
                     phase=-math.pi / 2 - 0.12, width=width, rho=densities[0], m=m,
                     axles=axles, scale=scale)
        _emit_rack(wb, meshes, n_rack=n_rack, module=m, pressure_angle=pa,
                   addendum=addendum, dedendum=dedendum, tooth_width=tooth_width,
                   backing=backing, z=z, width=width, rho=densities[1])
        if motor:
            ET.SubElement(model.section('actuator'), 'motor', name='drive_gear_a',
                          joint='gear_a_hinge', gear=mj.num(motor_gear), ctrlrange='-1 1')
        return model.to_xml()
    if standalone:
        reach = max(abs(x) + d['outer_radius'] for x, d in zip(xs, dims))
        model.standalone(timestep=timestep, floor=floor, floor_size=max(0.4, 4 * reach),
                         light_pos=(0.2, -0.35, 0.35), solver='PGS')
    wb = model.section('worldbody')
    for k, (d, x, phase, rho) in enumerate(zip(dims, xs, phases, densities)):
        name = f'gear_{"ab"[k]}'
        outline = _outline(d, phase)
        if axles:
            height = z + width * 1.25 / 2 + 0.004 * scale
            ET.SubElement(wb, 'geom', {'name': f'{name}_axle', 'type': 'cylinder',
                                       'class': mj.VISUAL, 'rgba': AXLE_RGBA,
                                       'size': mj.vec((0.45 * m, height / 2)),
                                       'pos': mj.vec((x, 0, height / 2))})
        body = ET.SubElement(wb, 'body', name=name, childclass=mj.CLASS, pos=mj.vec((x, 0, z)))
        ET.SubElement(body, 'joint', name=f'{name}_hinge', type='hinge', axis='0 0 1')
        visual = meshes.solid(f'{name}_visual', [(outline, [], -width / 2, width / 2)])
        ET.SubElement(body, 'geom', {'name': f'{name}_visual', 'type': 'mesh', 'mesh': visual,
                                     'class': mj.VISUAL, 'rgba': RGBA[k]})
        ET.SubElement(body, 'geom', {'name': f'{name}_hub', 'type': 'cylinder',
                                     'class': mj.VISUAL, 'rgba': HUB_RGBA,
                                     'size': mj.vec((0.35 * d['root_radius'], width * 1.25 / 2))})
        for i, piece in enumerate(_pieces(outline, d['root_radius'])):
            mesh = meshes.prism(f'{name}_col_{i}', piece, -width / 2, width / 2)
            ET.SubElement(body, 'geom', {'name': mesh, 'type': 'mesh', 'mesh': mesh,
                                         'class': mj.COLLISION, 'density': mj.num(rho)})
    if motor:
        ET.SubElement(model.section('actuator'), 'motor', name='drive_gear_a',
                      joint='gear_a_hinge', gear=mj.num(motor_gear), ctrlrange='-1 1')
    return model.to_xml()


def _dimensions(teeth: int, module: float, pressure_angle: float, addendum: float,
                dedendum: float, tooth_width: float) -> dict[str, float]:
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
    root_radius = pitch_radius - dedendum * module
    if root_radius <= module * 0.25:
        raise ValueError(f"The root circle of a {teeth}-tooth gear is too small; "
                         "use more teeth or a smaller dedendum.")
    return {
        'teeth': teeth,
        'pressure_angle': pressure_angle,
        'pitch_radius': pitch_radius,
        'base_radius': pitch_radius * math.cos(pressure_angle),
        'root_radius': root_radius,
        'outer_radius': pitch_radius + addendum * module,
        'half_tooth_angle_pitch': math.pi / teeth * tooth_width,
    }


def _involute(a: float) -> float:
    return math.tan(a) - a


def _half_angle(radius: float, d: dict) -> float:
    """Half the angular thickness of a tooth at a radius (involute relation)."""
    base = d['base_radius']
    inv_pitch = _involute(d['pressure_angle'])
    if radius <= base:
        return d['half_tooth_angle_pitch'] + inv_pitch
    alpha = math.acos(max(-1.0, min(1.0, base / radius)))
    return d['half_tooth_angle_pitch'] + inv_pitch - _involute(alpha)


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


def _emit_pinion(wb, meshes, d, *, name, x, y, z, phase, width, rho, m, axles, scale):
    """One external spur gear on a z hinge, at (x, y, z)."""
    outline = _outline(d, phase)
    if axles:
        height = z + width * 1.25 / 2 + 0.004 * scale
        ET.SubElement(wb, 'geom', {'name': f'{name}_axle', 'type': 'cylinder',
                                   'class': mj.VISUAL, 'rgba': AXLE_RGBA,
                                   'size': mj.vec((0.45 * m, height / 2)),
                                   'pos': mj.vec((x, y, height / 2))})
    body = ET.SubElement(wb, 'body', name=name, childclass=mj.CLASS, pos=mj.vec((x, y, z)))
    ET.SubElement(body, 'joint', name=f'{name}_hinge', type='hinge', axis='0 0 1')
    visual = meshes.solid(f'{name}_visual', [(outline, [], -width / 2, width / 2)])
    ET.SubElement(body, 'geom', {'name': f'{name}_visual', 'type': 'mesh', 'mesh': visual,
                                 'class': mj.VISUAL, 'rgba': RGBA[0]})
    ET.SubElement(body, 'geom', {'name': f'{name}_hub', 'type': 'cylinder',
                                 'class': mj.VISUAL, 'rgba': HUB_RGBA,
                                 'size': mj.vec((0.35 * d['root_radius'], width * 1.25 / 2))})
    for i, piece in enumerate(_pieces(outline, d['root_radius'])):
        mesh = meshes.prism(f'{name}_col_{i}', piece, -width / 2, width / 2)
        ET.SubElement(body, 'geom', {'name': mesh, 'type': 'mesh', 'mesh': mesh,
                                     'class': mj.COLLISION, 'density': mj.num(rho)})


def _rack_tooth(center_x: float, module: float, pressure_angle: float, addendum: float,
                dedendum: float, tooth_width: float) -> np.ndarray:
    """CCW trapezoid of one rack tooth, pitch line at y = 0."""
    pitch = math.pi * module
    half = tooth_width * pitch / 2
    tip_y = addendum * module
    root_y = -dedendum * module
    tip_half = half - tip_y * math.tan(pressure_angle)
    root_half = half + (-root_y) * math.tan(pressure_angle)
    if tip_half <= 1e-12:
        raise ValueError("addendum and pressure_angle leave no rack tooth tip; "
                         "reduce addendum or pressure_angle.")
    return pg.ensure_ccw(np.array([
        [center_x - root_half, root_y],
        [center_x - tip_half, tip_y],
        [center_x + tip_half, tip_y],
        [center_x + root_half, root_y],
    ]))


def _emit_rack(wb, meshes, *, n_rack, module, pressure_angle, addendum, dedendum,
               tooth_width, backing, z, width, rho):
    """A linear rack on a slide along +x, pitch line at y = 0."""
    pitch = math.pi * module
    centers = [(i + 0.5 - n_rack / 2) * pitch for i in range(n_rack)]
    teeth = [_rack_tooth(x, module, pressure_angle, addendum, dedendum, tooth_width)
             for x in centers]
    root_y = -dedendum * module
    bottom = root_y - backing
    left, right = centers[0] - pitch / 2, centers[-1] + pitch / 2
    bar = pg.ensure_ccw(np.array([
        [left, bottom], [right, bottom], [right, root_y], [left, root_y],
    ]))
    body = ET.SubElement(wb, 'body', name='rack', childclass=mj.CLASS, pos=mj.vec((0, 0, z)))
    ET.SubElement(body, 'joint', name='rack_slide', type='slide', axis='1 0 0')
    solids = [(bar, [], -width / 2, width / 2)]
    solids += [(t, [], -width / 2, width / 2) for t in teeth]
    visual = meshes.solid('rack_visual', solids)
    ET.SubElement(body, 'geom', {'name': 'rack_visual', 'type': 'mesh', 'mesh': visual,
                                 'class': mj.VISUAL, 'rgba': RGBA[1]})
    mesh = meshes.prism('rack_bar', bar, -width / 2, width / 2)
    ET.SubElement(body, 'geom', {'name': mesh, 'type': 'mesh', 'mesh': mesh,
                                 'class': mj.COLLISION, 'density': mj.num(rho)})
    for i, t in enumerate(teeth):
        mesh = meshes.prism(f'rack_tooth_{i}', t, -width / 2, width / 2)
        ET.SubElement(body, 'geom', {'name': mesh, 'type': 'mesh', 'mesh': mesh,
                                     'class': mj.COLLISION, 'density': mj.num(rho)})
