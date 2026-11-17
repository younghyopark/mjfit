"""Small MJCF writer shared by the generators."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from . import _polygon as pg

# Stiff, nearly hard contact: a 2 ms time constant needs a time step of 1 ms
# or less. These values are what the collision benchmark uses.
SOLREF = (0.002, 1.0)
SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)

CLASS = 'mjfit'                      # contact parameters for every part
VISUAL = 'mjfit_visual'              # render-only geoms
COLLISION = 'mjfit_collision'        # collision-only geoms, hidden by default

_SECTIONS = ('compiler', 'option', 'default', 'asset', 'worldbody', 'actuator')


def num(x: float) -> str:
    """Shortest text that reads back as the same float64."""
    return repr(float(x))


def vec(xs) -> str:
    return ' '.join(num(x) for x in np.ravel(xs))


def friction_triple(friction) -> tuple[float, float, float]:
    """A scalar sliding friction, or the full (sliding, torsional, rolling) triple."""
    if np.ndim(friction) == 0:
        return (float(friction), 0.01, 0.001)
    triple = tuple(float(f) for f in friction)
    if len(triple) != 3:
        raise ValueError(f"friction must be a number or three numbers, got {friction!r}.")
    return triple


class Model:
    """An MJCF document assembled section by section."""

    def __init__(self, name: str):
        self.root = ET.Element('mujoco', model=name)
        self.sections: dict[str, ET.Element] = {}

    def section(self, tag: str) -> ET.Element:
        if tag not in self.sections:
            el = ET.Element(tag)
            order = _SECTIONS.index(tag)
            index = sum(1 for t in self.sections if _SECTIONS.index(t) < order)
            self.root.insert(index, el)
            self.sections[tag] = el
        return self.sections[tag]

    def contact_defaults(self, *, friction, condim: int,
                         visual_classes: bool = False) -> ET.Element:
        """Contact parameters for every generated geom, in class `mjfit`."""
        if condim not in (1, 3, 4, 6):
            raise ValueError(f"condim must be 1, 3, 4 or 6, got {condim!r}.")
        top = ET.SubElement(self.section('default'), 'default', {'class': CLASS})
        ET.SubElement(top, 'geom', solref=vec(SOLREF), solimp=vec(SOLIMP),
                      condim=str(condim), friction=vec(friction_triple(friction)))
        if visual_classes:
            vis = ET.SubElement(top, 'default', {'class': VISUAL})
            ET.SubElement(vis, 'geom', group='2', contype='0', conaffinity='0', density='0')
            col = ET.SubElement(top, 'default', {'class': COLLISION})
            ET.SubElement(col, 'geom', group='3')
        return top

    def standalone(self, *, timestep: float, floor: bool, floor_size: float,
                   light_pos, solver: str | None = None) -> None:
        """What a file needs to run on its own: solver options, light, floor."""
        attrs = {'timestep': num(timestep), 'integrator': 'implicitfast'}
        if solver:
            attrs['solver'] = solver
        ET.SubElement(self.section('option'), 'option', attrs)
        wb = self.section('worldbody')
        light_pos = np.asarray(light_pos, dtype=float)
        direction = -light_pos / np.linalg.norm(light_pos)
        ET.SubElement(wb, 'light', pos=vec(light_pos), dir=vec(direction), diffuse='0.8 0.8 0.8')
        if floor:
            ET.SubElement(wb, 'geom', {'name': 'floor', 'type': 'plane', 'class': CLASS,
                                       'size': f'{num(floor_size)} {num(floor_size)} 0.01',
                                       'rgba': '0.92 0.92 0.92 1'})

    def to_xml(self) -> str:
        # <option> and <compiler> are single elements, not containers.
        for tag in ('option', 'compiler'):
            if tag in self.sections:
                wrapper = self.sections[tag]
                index = list(self.root).index(wrapper)
                self.root.remove(wrapper)
                for k, child in enumerate(list(wrapper)):
                    self.root.insert(index + k, child)
                del self.sections[tag]
        ET.indent(self.root, space='  ')
        return ET.tostring(self.root, encoding='unicode') + '\n'


class Meshes:
    """Mesh assets, inline or written to OBJ files."""

    def __init__(self, model: Model, mesh_dir: str | Path | None):
        self.asset = model.section('asset')
        self.dir = None if mesh_dir is None else Path(mesh_dir)
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
            ET.SubElement(model.section('compiler'), 'compiler', meshdir=self.dir.name)

    @property
    def exporting(self) -> bool:
        return self.dir is not None

    def prism(self, name: str, poly: np.ndarray, z0: float, z1: float) -> str:
        """A convex polygon extruded from z0 to z1 (MuJoCo hulls it)."""
        if self.dir is not None:
            write_obj(self.dir / f'{name}.obj', [(poly, [], z0, z1)])
            ET.SubElement(self.asset, 'mesh', name=name, file=f'{name}.obj')
        else:
            ring = np.column_stack([poly, np.zeros(len(poly))])
            lo, hi = ring.copy(), ring.copy()
            lo[:, 2], hi[:, 2] = z0, z1
            ET.SubElement(self.asset, 'mesh', name=name, vertex=vec(np.vstack([lo, hi])))
        return name

    def loft(self, name: str, bottom: np.ndarray, z0: float, top: np.ndarray, z1: float) -> str:
        """Two convex polygons with matching vertices, at z0 and z1, and the
        flat faces between them (MuJoCo hulls it). A prism when they agree."""
        if self.dir is not None:
            write_obj_faces(self.dir / f'{name}.obj',
                            *layered_triangles([([(bottom, z0), (top, z1)], [])]))
            ET.SubElement(self.asset, 'mesh', name=name, file=f'{name}.obj')
        else:
            lo = np.column_stack([bottom, np.full(len(bottom), z0)])
            hi = np.column_stack([top, np.full(len(top), z1)])
            ET.SubElement(self.asset, 'mesh', name=name, vertex=vec(np.vstack([lo, hi])))
        return name

    def layered(self, name: str, solids) -> str:
        """A visual mesh like `solid`, for solids whose rings change between
        levels (chamfers): see `layered_triangles`."""
        verts, faces = layered_triangles(solids)
        if self.dir is not None:
            write_obj_faces(self.dir / f'{name}.obj', verts, faces)
            ET.SubElement(self.asset, 'mesh', name=name, file=f'{name}.obj')
        else:
            ET.SubElement(self.asset, 'mesh', name=name, vertex=vec(verts),
                          face=' '.join(str(int(i)) for i in np.ravel(faces)))
        return name

    def points(self, name: str, vertices: np.ndarray, faces=None) -> str:
        """A mesh from 3D points: their convex hull, or the given triangles."""
        if self.dir is not None:
            if faces is None:
                raise ValueError("Exporting a point mesh needs its faces.")
            write_obj_faces(self.dir / f'{name}.obj', vertices, faces)
            ET.SubElement(self.asset, 'mesh', name=name, file=f'{name}.obj')
        else:
            attrs = {'name': name, 'vertex': vec(vertices)}
            if faces is not None:
                attrs['face'] = ' '.join(str(int(i)) for i in np.ravel(faces))
            ET.SubElement(self.asset, 'mesh', attrs)
        return name

    def solid(self, name: str, solids) -> str:
        """A visual mesh: extruded polygons with holes, one watertight OBJ.

        Each solid is (outer, holes, z0, z1). Inline, the triangles are
        written as vertex and face lists.
        """
        if self.dir is not None:
            write_obj(self.dir / f'{name}.obj', solids)
            ET.SubElement(self.asset, 'mesh', name=name, file=f'{name}.obj')
            return name
        verts, faces = extrusion_triangles(solids)
        ET.SubElement(self.asset, 'mesh', name=name, vertex=vec(verts),
                      face=' '.join(str(int(i)) for i in np.ravel(faces)))
        return name


def extrusion_triangles(solids) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and outward-facing triangles of extruded polygons with holes."""
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    def add(x, y, z) -> int:
        verts.append((float(x), float(y), float(z)))
        return len(verts) - 1

    for outer, holes, z0, z1 in solids:
        for tri in pg.triangulate(outer, list(holes)):
            top = [add(p[0], p[1], z1) for p in tri]
            bot = [add(p[0], p[1], z0) for p in tri]
            faces.append((top[0], top[1], top[2]))
            faces.append((bot[0], bot[2], bot[1]))
        for ring, outward in [(outer, True)] + [(h, False) for h in holes]:
            n = len(ring)
            for i in range(n):
                v0, v1 = ring[i], ring[(i + 1) % n]
                b0, b1 = add(v0[0], v0[1], z0), add(v1[0], v1[1], z0)
                t1, t0 = add(v1[0], v1[1], z1), add(v0[0], v0[1], z1)
                if outward:
                    faces += [(b0, b1, t1), (b0, t1, t0)]
                else:
                    faces += [(b0, t1, b1), (b0, t0, t1)]
    return np.asarray(verts), np.asarray(faces)


def layered_triangles(solids) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and outward-facing triangles of solids whose rings vary with height.

    Each solid is (outer, holes): `outer` is a list of (polygon, z) levels from
    bottom to top, all with the same vertex count, and each hole is a list of
    levels of its own that starts and ends at the outer's first and last z.
    Walls join consecutive levels vertex to vertex; the caps are the bottom
    and top levels, triangulated with their holes.
    """
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    def add(x, y, z) -> int:
        verts.append((float(x), float(y), float(z)))
        return len(verts) - 1

    for outer, holes in solids:
        for k, flip in ((0, True), (-1, False)):
            ring, z = outer[k]
            for tri in pg.triangulate(ring, [h[k][0] for h in holes]):
                ids = [add(p[0], p[1], z) for p in tri]
                faces.append((ids[0], ids[2], ids[1]) if flip else (ids[0], ids[1], ids[2]))
        for levels, outward in [(outer, True)] + [(h, False) for h in holes]:
            for (r0, z0), (r1, z1) in zip(levels[:-1], levels[1:]):
                n = len(r0)
                for i in range(n):
                    j = (i + 1) % n
                    b0, b1 = add(r0[i, 0], r0[i, 1], z0), add(r0[j, 0], r0[j, 1], z0)
                    t1, t0 = add(r1[j, 0], r1[j, 1], z1), add(r1[i, 0], r1[i, 1], z1)
                    if outward:
                        faces += [(b0, b1, t1), (b0, t1, t0)]
                    else:
                        faces += [(b0, t1, b1), (b0, t0, t1)]
    return np.asarray(verts), np.asarray(faces)


def write_obj(path: Path, solids) -> None:
    """Write extruded polygons with holes as one watertight OBJ.

    The caps are triangulated as whole regions, so a decomposed part gets no
    interior walls between its convex pieces (they would z-fight in renders).
    """
    verts, faces = extrusion_triangles(solids)
    write_obj_faces(path, verts, faces)


def write_obj_faces(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    lines = ['# mjfit'] + [f'v {num(x)} {num(y)} {num(z)}' for x, y, z in verts]
    lines += [f'f {a + 1} {b + 1} {c + 1}' for a, b, c in faces]
    Path(path).write_text('\n'.join(lines) + '\n')
