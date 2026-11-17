"""Spur gears and meshing gear pairs."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from . import _mjcf as mj
from . import _polygon as pg
from . import shapes

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
    bore: float = 0.0,
    bore_chamfer: float = 0.0,
    shafts: bool = False,
    clearance: float = 1e-4,
    shaft_chamfer: float = 0.0,
    max_error: float = 1e-5,
    density: float = 7800.0,
    axles: bool = True,
    motor: bool | None = None,
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
    """A spur gear, or a pair of meshing spur gears, as an MJCF string.

    Each gear turns on a hinge about +z. Its collision shape is one convex
    prism per tooth plus one for the hub inside the root circle; a separate
    visual mesh renders the outline (the collision prisms are in geom group 3).
    With a ``bore`` the hub is a ring of convex wedges around it, and with
    ``shafts`` each gear sits free on a fixed shaft through its bore instead
    of on a hinge, to be slid on and off as in a gear assembly task.

    Hinged gears give MuJoCo's default Newton solver almost no contact
    regularization (their mass sits on the axis), so standalone files use the
    PGS solver, which meshes them at the exact ratio.

    Args:
        teeth: Tooth count of one gear, or two counts for a meshing pair.
        module: Pitch diameter per tooth, in meters; shared by both gears.
        face_width: Thickness of the gears.
        pressure_angle: In degrees.
        addendum: Tooth height above the pitch circle, in modules.
        dedendum: Depth below the pitch circle, in modules.
        tooth_width: Tooth thickness at the pitch circle, as a fraction of the
            circular pitch; below 0.5 leaves backlash.
        center_distance: Distance between the axes. Default: the sum of the
            pitch radii.
        bore: Diameter of a round hole through each gear's hub; 0 for a solid
            hub. Its polygon circumscribes the circle, so the bore is never
            smaller than drawn.
        bore_chamfer: Bevels the bore's edge on both faces at 45 degrees: the
            bore widens by this much at each face. 0 for sharp edges.
        shafts: Put each gear on a fixed shaft through its bore, on a free
            joint, instead of on a hinge. Needs a ``bore``.
        clearance: Gap between the shaft and the bore on every edge.
        shaft_chamfer: Bevels the shaft's top end at 45 degrees.
        max_error: How far the bore's polygon may stray from its circle.
        density: kg/m^3, or one per gear.
        axles: Draw the axles (visual only; with ``shafts`` the shafts are the
            axles).
        motor: Add a motor on the first gear's hinge (control in [-1, 1]).
            Default: on, unless the gears sit on ``shafts``.
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
        hinges ``gear_a_hinge`` (and ``gear_b_hinge``), or with ``shafts`` on
        free joints ``gear_a_joint`` (and ``gear_b_joint``) around fixed shafts
        ``gear_a_shaft`` (and ``gear_b_shaft``).
    """
    counts = [int(t) for t in np.atleast_1d(teeth)]
    if len(counts) not in (1, 2) or any(t != t_in for t, t_in in zip(counts, np.atleast_1d(teeth))):
        raise ValueError(f"teeth must be one or two integers, got {teeth!r}.")
    densities = [float(d) for d in np.broadcast_to(np.atleast_1d(density), (len(counts),))]
    if face_width <= 0 or scale <= 0:
        raise ValueError("face_width and scale must be positive.")
    for name, value in dict(bore=bore, bore_chamfer=bore_chamfer, clearance=clearance,
                            shaft_chamfer=shaft_chamfer).items():
        if not (math.isfinite(value) and value >= 0):
            raise ValueError(f"{name} must be a nonnegative number, got {value!r}.")
    if not max_error > 0:
        raise ValueError("max_error must be positive.")
    if bore == 0 and (bore_chamfer > 0 or shafts):
        raise ValueError("bore_chamfer and shafts need a bore.")
    if shaft_chamfer > 0 and not shafts:
        raise ValueError("shaft_chamfer needs shafts.")
    if motor is None:
        motor = not shafts
    elif motor and shafts:
        raise ValueError("The motor drives a hinge; gears on shafts have none. Set motor=False.")
    m, width = module * scale, face_width * scale
    bore_poly = mouth = shaft = shaft_tip = None
    if bore > 0:
        bore_poly = shapes.circle(bore * scale / 2, max_error=max_error * scale)
        ch = bore_chamfer * scale
        if not 2 * ch < width:
            raise ValueError("bore_chamfer must be smaller than half the face_width.")
        mouth = pg.offset_outward(bore_poly, ch) if ch > 0 else bore_poly
        if shafts:
            if max_error > clearance:
                raise ValueError("max_error must not exceed the clearance, or the shaft's "
                                 "corners could overlap the round bore it stands for.")
            shaft = pg.offset_inward(bore_poly, clearance * scale)
            if not pg.offset_is_valid(bore_poly, shaft):
                raise ValueError("clearance is too large for the bore.")
            shaft_tip = pg.offset_inward(shaft, shaft_chamfer * scale)
            if shaft_chamfer > 0 and not pg.offset_is_valid(shaft, shaft_tip):
                raise ValueError("shaft_chamfer is too large for the shaft.")
    dims = [_dimensions(t, m, math.radians(pressure_angle), addendum, dedendum, tooth_width)
            for t in counts]
    pair = len(dims) == 2
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
    if standalone:
        reach = max(abs(x) + d['outer_radius'] for x, d in zip(xs, dims))
        # Free gears resting on the floor touch it with every piece of the bored hub.
        model.standalone(timestep=timestep, floor=floor, floor_size=max(0.4, 4 * reach),
                         light_pos=(0.2, -0.35, 0.35), solver='PGS',
                         memory='256M' if shaft is not None else None)
    wb = model.section('worldbody')
    for k, (d, x, phase, rho) in enumerate(zip(dims, xs, phases, densities)):
        name = f'gear_{"ab"[k]}'
        outline = _outline(d, phase)
        height = z + width * 1.25 / 2 + 0.004 * scale
        if shaft is not None:
            ch = shaft_chamfer * scale
            pos = mj.vec((x, 0, 0))
            col = meshes.prism(f'{name}_shaft', shaft, 0.0, height - ch)
            ET.SubElement(wb, 'geom', {'name': f'{name}_shaft', 'type': 'mesh', 'mesh': col,
                                       'class': mj.CLASS, 'rgba': AXLE_RGBA, 'pos': pos})
            if ch > 0:
                tip = meshes.loft(f'{name}_shaft_tip', shaft, height - ch, shaft_tip, height)
                ET.SubElement(wb, 'geom', {'name': f'{name}_shaft_tip', 'type': 'mesh',
                                           'mesh': tip, 'class': mj.CLASS, 'rgba': AXLE_RGBA,
                                           'pos': pos})
        elif axles:
            ET.SubElement(wb, 'geom', {'name': f'{name}_axle', 'type': 'cylinder',
                                       'class': mj.VISUAL, 'rgba': AXLE_RGBA,
                                       'size': mj.vec((0.45 * m, height / 2)),
                                       'pos': mj.vec((x, 0, height / 2))})
        body = ET.SubElement(wb, 'body', name=name, childclass=mj.CLASS, pos=mj.vec((x, 0, z)))
        if shaft is not None:
            ET.SubElement(body, 'freejoint', name=f'{name}_joint')
        else:
            ET.SubElement(body, 'joint', name=f'{name}_hinge', type='hinge', axis='0 0 1')
        if bore_poly is None:
            visual = meshes.solid(f'{name}_visual', [(outline, [], -width / 2, width / 2)])
        else:
            h = width / 2 - bore_chamfer * scale
            visual = meshes.layered(f'{name}_visual', [(
                [(outline, -width / 2), (outline, width / 2)],
                [[(mouth, -width / 2), (bore_poly, -h), (bore_poly, h), (mouth, width / 2)]])])
        ET.SubElement(body, 'geom', {'name': f'{name}_visual', 'type': 'mesh', 'mesh': visual,
                                     'class': mj.VISUAL, 'rgba': RGBA[k]})
        if bore_poly is None:
            ET.SubElement(body, 'geom', {'name': f'{name}_hub', 'type': 'cylinder',
                                         'class': mj.VISUAL, 'rgba': HUB_RGBA,
                                         'size': mj.vec((0.35 * d['root_radius'],
                                                         width * 1.25 / 2))})
        pieces = _pieces(outline, d['root_radius'], bore_poly, mouth, width,
                         bore_chamfer * scale)
        for i, (p0, z0, p1, z1) in enumerate(pieces):
            name_i = f'{name}_col_{i}'
            mesh = (meshes.prism(name_i, p0, z0, z1) if p1 is p0
                    else meshes.loft(name_i, p0, z0, p1, z1))
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


def _pieces(outline: np.ndarray, root_radius: float, bore=None, mouth=None,
            width: float = 0.0, chamfer: float = 0.0):
    """Convex pieces as (bottom polygon, z0, top polygon, z1), the same polygon
    object at both ends for a prism.

    Without a bore: the hub polygon through every root-circle point and one
    piece per tooth between consecutive root points, or a fan around the
    center if a tooth is not convex. With a bore: the teeth, and the hub cut
    into wedges along the bisectors of the bore's corners; a chamfer adds a
    band at each face whose inner edge runs from the mouth to the bore.
    """
    h = width / 2
    hub, teeth = _hub_and_teeth(outline, root_radius)
    if bore is None:
        if all(pg.is_convex(t) for t in teeth) and pg.is_convex(hub):
            return [(p, -h, p, h) for p in [hub] + teeth]
        center = np.zeros(2)
        n = len(outline)
        fan = [np.array([center, outline[i], outline[(i + 1) % n]]) for i in range(n)]
        return [(t, -h, t, h) for t in fan if abs(pg.signed_area(t)) > 1e-14]
    if not (pg.is_convex(hub) and all(pg.is_convex(t) for t in teeth)):
        raise ValueError("A bore needs convex teeth and hub; this tooth setting is not supported.")
    if pg.min_rim(hub, mouth) <= 1e-10:
        raise ValueError("The bore (with its chamfer) does not fit inside the root circle.")
    out = [(t, -h, t, h) for t in teeth]
    ring = pg.bisector_wedges(bore, hub)
    if chamfer == 0:
        return out + [(w, -h, w, h) for w in ring]
    rim = pg.bisector_wedges(mouth, hub)
    lip = h - chamfer
    for w, r in zip(ring, rim):
        out += [(r, -h, w, -lip), (w, -lip, w, lip), (w, lip, r, h)]
    return out


def _hub_and_teeth(outline: np.ndarray, root_radius: float):
    """The polygon through every root-circle point, and each tooth's outline
    between consecutive root points."""
    on_root = np.abs(np.linalg.norm(outline, axis=1) - root_radius) < 1e-9 * max(root_radius, 1e-3)
    roots = np.flatnonzero(on_root)
    hub = outline[roots]
    teeth = []
    for a, b in zip(roots, np.roll(roots, -1)):
        run = outline[a:b + 1] if b > a else np.vstack([outline[a:], outline[:b + 1]])
        if len(run) > 2:
            teeth.append(run)
    return hub, teeth
