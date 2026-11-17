"""Peg-in-hole geometry: exact convex pieces, clearance, frames and masses."""

import warnings
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from conftest import hull_volume, inline_meshes, prism_volume

import mjfit
from mjfit import _polygon as pg
from mjfit import shapes

CONTROL = np.array([[-0.022, -0.013], [0.024, -0.010], [0.019, 0.017], [-0.017, 0.022]])
CASES = {
    'circle': dict(hole=shapes.circle(0.008)),
    'star': dict(hole=shapes.star(5, 0.012, 0.0054)),
    'notched': dict(
        hole=[
            [-0.006, -0.003],
            [0.006, -0.003],
            [0.006, 0.003],
            [0.0015, 0.003],
            [0.0015, 0.002],
            [-0.0015, 0.002],
            [-0.0015, 0.003],
            [-0.006, 0.003],
        ]
    ),
    'star_islands': dict(
        hole=shapes.star(6, 0.02, 0.013),
        islands=[
            shapes.circle(0.002, center=(0.004, 0)),
            shapes.rect(0.003, 0.002, center=(-0.004, 0.001)),
        ],
        plate_thickness=0.005,
        peg_length=0.012,
    ),
    'handle': dict(hole=shapes.stadium(0.03, 0.012), handle_height=0.02, handle_margin=0.004),
    'plate': dict(
        peg=shapes.bezier(CONTROL, [0.5, 0, 0.25, 0.5]),
        handle=shapes.bezier(CONTROL * 1.8, 0.5),
        handle_height=0.025,
        plate=shapes.bezier(CONTROL * 3, [0.5, 0.3, 0, 0.4]),
        min_rim=0.005,
        bottom_thickness=0.002,
        peg_mass=0.55,
        clearance=0.001,
        peg_length=0.045,
        center=False,
    ),
}


@pytest.mark.parametrize('case', CASES)
def test_every_piece_is_an_exact_convex_prism(case):
    """MuJoCo collides each mesh's convex hull; for a convex prism the hull
    is the prism itself, so the hull's volume equals the prism's."""
    xml = mjfit.peg(**CASES[case])
    model = mujoco.MjModel.from_xml_string(xml)
    for name, verts in inline_meshes(xml).items():
        assert hull_volume(model, model.mesh(name).id) == pytest.approx(
            prism_volume(verts), rel=1e-6
        ), name


@pytest.mark.parametrize('case', CASES)
def test_pieces_tile_the_plate_and_the_peg(case):
    kw = CASES[case]
    xml = mjfit.peg(**kw)
    meshes = inline_meshes(xml)
    area = {}
    for name, v in meshes.items():
        kind = name.rsplit('_', 1)[0] if name[-1].isdigit() else name
        ring = v[: len(v) // 2, :2]
        assert pg.is_simple(ring) and pg.signed_area(ring) > 0, (
            f'{name} is not a CCW simple polygon'
        )
        if kind in ('plate', 'island', 'peg', 'peg_ring', 'peg_back'):
            area[kind] = area.get(kind, 0.0) + pg.signed_area(ring)
    # The peg's solid back covers the whole peg outline, and the ring the
    # outline less the pockets.
    if 'peg_back' in area:
        assert area['peg_back'] == pytest.approx(
            area['peg_ring']
            + sum(
                pg.signed_area(pg.offset_outward(pg.ensure_ccw(np.asarray(i, float)), 1e-4))
                for i in kw['islands']
            ),
            rel=1e-9,
        )


def test_clearance_is_exact_on_every_edge():
    hole = shapes.star(5, 0.012, 0.0054)
    for c in (2e-5, 1e-4, 5e-4):
        xml = mjfit.peg(hole=hole, clearance=c, center=False)
        peg = np.vstack(
            [v[: len(v) // 2, :2] for k, v in inline_meshes(xml).items() if k.startswith('peg_')]
        )
        outline = pg.offset_inward(hole, c)
        # every peg edge lies on the hole edge's line moved in by c
        edges = np.roll(hole, -1, axis=0) - hole
        inward = (
            np.column_stack([-edges[:, 1], edges[:, 0]]) / np.linalg.norm(edges, axis=1)[:, None]
        )
        dist = np.einsum('ij,ij->i', inward, outline - hole)
        np.testing.assert_allclose(dist, c, rtol=1e-9)
        assert set(map(tuple, np.round(outline, 12))) <= set(map(tuple, np.round(peg, 12)))


def test_circle_hole_is_never_smaller_than_drawn():
    radius, error = 0.008, 1e-5
    hole = shapes.circle(radius, max_error=error)
    edges = np.roll(hole, -1, axis=0) - hole
    mid = hole + edges / 2
    np.testing.assert_allclose(np.linalg.norm(mid, axis=1), radius, rtol=1e-12)  # flats touch
    assert np.linalg.norm(hole, axis=1).max() <= radius + error  # corners bounded


def test_peg_corners_stay_inside_the_round_hole():
    """Against a round peg the peg's corners stick out by the hole's corner
    error times 1 - clearance/radius, so with max_error at most the clearance
    they stay inside the round hole."""
    radius, c = 0.00128, 5e-5
    for error in (1e-5, 5e-5):
        spec = {'type': 'circle', 'radius': radius, 'max_error': error}
        xml = mjfit.peg(hole=spec, clearance=c, center=False)
        reach = np.linalg.norm(inline_meshes(xml)['peg_0'][:, :2], axis=1).max()
        n = len(shapes.from_spec(spec))
        assert reach == pytest.approx((radius - c) / np.cos(np.pi / n), rel=1e-9)
        assert reach < radius


def test_curves_coarser_than_the_clearance_warn():
    usb_c = {'type': 'stadium', 'width': 0.00834, 'height': 0.00256}
    tongue = {'type': 'stadium', 'width': 0.0065, 'height': 0.0007}
    rounded = {'type': 'bezier', 'vertices': CONTROL.tolist()}
    for kw in (
        dict(hole=usb_c),
        dict(peg=usb_c),
        dict(hole=usb_c, scale=10.0),
        dict(hole=shapes.rect(0.009, 0.003), islands=[tongue]),
        dict(hole=rounded),
    ):
        with pytest.warns(UserWarning, match='more than the 5e-05 m clearance'):
            mjfit.peg(**kw, clearance=5e-5)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        mjfit.peg(hole={**usb_c, 'max_error': 1e-5}, clearance=5e-5)


def test_frames_mass_and_bottom():
    kw = CASES['plate']
    xml = mjfit.peg(**kw, floor=False, spawn_height=-0.03)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.body('peg').mass[0] == pytest.approx(0.55, abs=1e-12)
    # geoms of one material: mass over volume is the same everywhere
    geoms = ET.fromstring(xml).findall(".//body[@name='peg']/geom")
    meshes = inline_meshes(xml)
    density = {
        g.get('name'): float(g.get('mass')) / prism_volume(meshes[g.get('mesh')]) for g in geoms
    }
    assert max(density.values()) == pytest.approx(min(density.values()), rel=1e-9)
    # the plate stands on z = 0 and its bottom closes the hole
    bottom = model.geom('bottom')
    z = data.geom_xpos[bottom.id][2]
    assert z == pytest.approx(0.001, abs=1e-8)  # 2 mm bottom centered at 1 mm (float32 mesh frame)
    plate_top = 0.002 + 0.01
    assert data.xpos[model.body('plate').id][2] == pytest.approx(0.002 + 0.005)
    peg_z = data.xpos[model.body('peg').id][2]
    assert peg_z == pytest.approx(plate_top - 0.03 + 0.045 / 2)


def test_centering_moves_the_hole_centroid_to_the_origin():
    hole = shapes.rect(0.01, 0.006, center=(0.02, -0.01), rotate=30)
    xml = mjfit.peg(hole=hole, include_peg=False)
    ring = np.vstack([v[: len(v) // 2, :2] for v in inline_meshes(xml).values()])
    assert np.abs(ring.mean(axis=0)).max() < 1e-12


def test_scale_scales_every_length():
    a = mjfit.peg(hole=shapes.circle(0.008), clearance=1e-4)
    b = mjfit.peg(hole=shapes.circle(0.008), clearance=1e-4, scale=3.0)
    ma, mb = inline_meshes(a), inline_meshes(b)
    assert ma.keys() == mb.keys()
    for k in ma:
        np.testing.assert_allclose(mb[k], 3 * ma[k], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize(
    'kw,match',
    [
        (dict(hole=shapes.rect(0.01, 0.005), clearance=0.003), 'too large'),
        (dict(hole=shapes.star(5, 0.012, 0.0054), clearance=0.006), 'too large'),
        (dict(hole=shapes.rect(0.01, 0.005), clearance=-1e-4), 'nonnegative'),
        (dict(hole=[[0, 0], [1, 1], [1, 0], [0, 1]]), 'crosses itself'),
        (dict(hole=[[0, 0], [0.01, 0], [0.01, 0], [0, 0.01]]), 'repeats vertex'),
        (
            dict(hole=shapes.rect(0.01, 0.005), islands=[shapes.circle(0.001, center=(0.02, 0))]),
            'not inside',
        ),
        (
            dict(
                hole=shapes.rect(0.02, 0.01),
                islands=[shapes.circle(0.002), shapes.circle(0.002, center=(0.0041, 0))],
            ),
            'closer than twice',
        ),
        (dict(hole=shapes.rect(0.01, 0.005), plate_size=0.009), 'no rim'),
        (
            dict(
                hole=shapes.rect(0.01, 0.005), handle=shapes.rect(0.005, 0.002), handle_height=0.01
            ),
            'contain',
        ),
        (dict(peg=shapes.rect(0.01, 0.005), islands=[shapes.circle(0.001)]), 'need a hole'),
        (dict(hole=shapes.rect(0.01, 0.005), plate_thickness=0), 'positive'),
    ],
)
def test_invalid_input_is_rejected_with_a_reason(kw, match):
    with pytest.raises(ValueError, match=match):
        mjfit.peg(**kw)


def test_collinear_pockets_leave_no_flat_triangle():
    """earcut can join three pocket corners on one line, a triangle MuJoCo
    cannot hull. db9's pins at 0.4 mm radius do; the flat triangle is dropped."""
    cfg = mjfit.load('db9')
    islands = [{**s, 'radius': 0.0004} for s in cfg['islands']]
    mujoco.MjModel.from_xml_string(mjfit.build(cfg, islands=islands))


def test_hollow_peg_back_is_decomposed_when_not_convex():
    xml = mjfit.peg(**CASES['star_islands'])
    backs = [k for k in inline_meshes(xml) if k.startswith('peg_back_')]
    assert len(backs) > 1


def test_standalone_false_keeps_contact_parameters():
    xml = mjfit.peg(hole=shapes.circle(0.008), standalone=False)
    root = ET.fromstring(xml)
    assert root.find('option') is None and root.find(".//geom[@name='floor']") is None
    assert root.find("./default/default[@class='mjfit']/geom").get('solref') == '0.002 1.0'


def test_mesh_export_loads_and_matches(tmp_path):
    kw = CASES['star_islands']
    path = tmp_path / 'star.xml'
    path.write_text(mjfit.peg(**kw, mesh_dir=tmp_path / 'star_meshes'))
    exported = mujoco.MjModel.from_xml_path(str(path))
    inline = mujoco.MjModel.from_xml_string(mjfit.peg(**kw))
    assert exported.body('peg').mass[0] == pytest.approx(inline.body('peg').mass[0], rel=1e-6)
    assert (tmp_path / 'star_meshes' / 'plate_visual.obj').exists()


# ----- chamfers ---------------------------------------------------------------

CHAMFERED = {
    'round': dict(hole=shapes.circle(0.0045, max_error=1e-5), clearance=5e-4,
                  peg_chamfer=5e-4, hole_chamfer=1e-3, peg_length=0.05, plate_thickness=0.025),
    'rect': dict(hole=shapes.rect(0.016, 0.010), clearance=1e-4, peg_chamfer=1.6e-3),
    'hole_only': dict(hole=shapes.rect(0.012, 0.008), clearance=1e-4, hole_chamfer=1e-3),
    'plate': dict(peg=shapes.bezier(CONTROL, 0.5), clearance=1e-4, peg_chamfer=1e-3,
                  hole_chamfer=1.5e-3, plate=shapes.bezier(CONTROL * 3, 0.4), center=False,
                  handle_height=0.01, peg_mass=0.2),
}


def _loft_volume(v: np.ndarray) -> float:
    """Volume of a loft given as a bottom ring followed by a matching top ring."""
    n = len(v) // 2
    lo, hi = v[:n, :2], v[n:, :2]
    area = [abs(pg.signed_area(r)) for r in (lo, (lo + hi) / 2, hi)]
    return (v[n:, 2].mean() - v[:n, 2].mean()) / 6 * (area[0] + 4 * area[1] + area[2])


def _volume(model: mujoco.MjModel, body: str) -> float:
    b = model.body(body).id
    return sum(hull_volume(model, int(model.geom_dataid[g]))
               for g in range(model.ngeom) if model.geom_bodyid[g] == b)


@pytest.mark.parametrize('case', CHAMFERED)
def test_chamfered_pieces_are_exact_convex_lofts(case):
    """Each chamfered piece is two rings joined by flat faces; MuJoCo's hull
    of its points has exactly the loft's volume, so nothing bulges."""
    xml = mjfit.peg(**CHAMFERED[case])
    model = mujoco.MjModel.from_xml_string(xml)
    for name, verts in inline_meshes(xml).items():
        assert hull_volume(model, model.mesh(name).id) == pytest.approx(
            _loft_volume(verts), rel=1e-6), name


def _frustum(poly: np.ndarray, grow: float, height: float) -> float:
    """Volume of a 45-degree band from `poly` to `poly` moved out by `grow`."""
    return _loft_volume(np.vstack([np.column_stack([poly, np.zeros(len(poly))]),
                                   np.column_stack([pg.offset_outward(poly, grow),
                                                    np.full(len(poly), height)])]))


@pytest.mark.parametrize('case', ['round', 'rect', 'hole_only'])
def test_chamfered_parts_have_the_exact_volume(case):
    """The pieces tile the chamfered plate and peg without gaps or overlaps."""
    kw = {'peg_length': 0.03, 'plate_thickness': 0.01, 'peg_chamfer': 0.0, 'hole_chamfer': 0.0,
          **CHAMFERED[case]}
    model = mujoco.MjModel.from_xml_string(mjfit.peg(**kw))
    hole = pg.ensure_ccw(np.asarray(kw['hole'], float))
    hole = hole - mjfit._peg._centroid(hole)
    insert = pg.offset_inward(hole, kw['clearance'])
    t, h = kw['plate_thickness'], kw['hole_chamfer']
    reach = float(np.abs(hole).max())
    side = 2 * (reach + max(0.005, reach))
    plate = side**2 * t - abs(pg.signed_area(hole)) * (t - h) - _frustum(hole, h, h)
    length, p = kw['peg_length'], kw['peg_chamfer']
    tip = pg.offset_inward(insert, p)
    peg = abs(pg.signed_area(insert)) * (length - p) + _frustum(tip, p, p)
    # MuJoCo keeps mesh vertices in float32.
    assert _volume(model, 'plate') == pytest.approx(plate, rel=1e-6)
    assert _volume(model, 'peg') == pytest.approx(peg, rel=1e-6)


def test_chamfers_are_45_degrees_and_keep_the_clearance():
    hole = shapes.rect(0.016, 0.010)
    c, p, h, t = 1e-4, 1.6e-3, 1e-3, 0.01
    meshes = inline_meshes(mjfit.peg(hole=hole, clearance=c, peg_chamfer=p, hole_chamfer=h,
                                     plate_thickness=t, center=False))
    insert = pg.offset_inward(hole, c)
    tip = meshes['peg_tip']
    n = len(tip) // 2
    np.testing.assert_allclose(tip[:n, :2], pg.offset_inward(insert, p), atol=1e-15)
    np.testing.assert_allclose(tip[n:, :2], insert, atol=1e-15)
    assert tip[n:, 2].mean() - tip[:n, 2].mean() == pytest.approx(p, rel=1e-12)
    # The mouth's top ring holds the hole moved out by h, h above its bottom ring.
    mouth = pg.offset_outward(hole, h)
    tops = np.vstack([v[len(v) // 2:] for k, v in meshes.items() if k.startswith('plate_mouth_')])
    bots = np.vstack([v[:len(v) // 2] for k, v in meshes.items() if k.startswith('plate_mouth_')])
    assert tops[:, 2].min() - bots[:, 2].max() == pytest.approx(h, rel=1e-12)
    for corner in mouth:
        assert np.min(np.linalg.norm(tops[:, :2] - corner, axis=1)) < 1e-15
    # Below the mouth the straight plate pieces still hold the hole exactly.
    for corner in hole:
        lower = np.vstack([v[:, :2] for k, v in meshes.items()
                           if k.startswith('plate_') and not k.startswith('plate_mouth_')])
        assert np.min(np.linalg.norm(lower - corner, axis=1)) < 1e-15


def test_chamfered_peg_slides_into_the_hole():
    """A peg dropped off-centre onto the mouth is guided in by the chamfers; with
    sharp edges it lands on the plate's top face."""
    def drop(chamfer):
        xml = mjfit.peg(hole=shapes.circle(0.006, max_error=1e-5), clearance=2e-4,
                        peg_chamfer=chamfer, hole_chamfer=chamfer, spawn_height=0.002,
                        peg_length=0.03, plate_thickness=0.01)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.qpos[model.joint('peg_joint').qposadr[0]] += 0.0012     # 6 clearances off-axis
        mujoco.mj_step(model, data, nstep=1500)
        tip = data.body('peg').xpos[2] - 0.015
        return tip - model.body('plate').pos[2] - 0.005              # below the plate top
    assert drop(1.5e-3) < -0.005
    assert drop(0.0) > -1e-3                                      # rests on the rim


@pytest.mark.parametrize('kw,match', [
    (dict(hole=shapes.star(5, 0.012, 0.0054), hole_chamfer=1e-3), 'convex hole'),
    (dict(peg=shapes.star(5, 0.012, 0.0054), peg_chamfer=1e-3), 'convex peg'),
    (dict(hole=shapes.rect(0.01, 0.005), hole_chamfer=0.01), 'plate_thickness'),
    (dict(hole=shapes.rect(0.01, 0.005), peg_chamfer=0.03), 'peg_length'),
    (dict(hole=shapes.rect(0.01, 0.005), peg_chamfer=0.003), 'too large'),
    (dict(hole=shapes.rect(0.01, 0.005), hole_chamfer=0.006, plate_size=0.02), 'no rim'),
    (dict(hole=shapes.rect(0.02, 0.01), islands=[shapes.circle(0.002)], peg_chamfer=1e-3),
     'islands'),
    (dict(hole=shapes.rect(0.01, 0.005), peg_chamfer=-1e-3), 'nonnegative'),
])
def test_invalid_chamfers_are_rejected_with_a_reason(kw, match):
    with pytest.raises(ValueError, match=match):
        mjfit.peg(**kw)


def test_chamfered_mesh_export_is_watertight_and_matches(tmp_path):
    kw = CHAMFERED['plate']
    path = tmp_path / 'chamfered.xml'
    path.write_text(mjfit.peg(**kw, mesh_dir=tmp_path / 'meshes'))
    exported = mujoco.MjModel.from_xml_path(str(path))
    inline = mujoco.MjModel.from_xml_string(mjfit.peg(**kw))
    assert exported.body('peg').mass[0] == pytest.approx(0.2, rel=1e-9)
    assert inline.body('peg').mass[0] == pytest.approx(0.2, rel=1e-9)
    for name in ('plate_visual', 'peg_visual'):
        lines = (tmp_path / 'meshes' / f'{name}.obj').read_text().splitlines()
        verts = np.array([list(map(float, ln.split()[1:])) for ln in lines if ln.startswith('v ')])
        faces = np.array([list(map(int, ln.split()[1:])) for ln in lines
                          if ln.startswith('f ')]) - 1
        keys = {tuple(np.round(v, 12)): k for k, v in enumerate(verts)}
        ids = np.array([keys[tuple(np.round(v, 12))] for v in verts])[faces]
        edges = {}
        for a, b, c in ids:
            for e in ((a, b), (b, c), (c, a)):
                edges[e] = edges.get(e, 0) + 1
        # Every directed edge is matched by its reverse exactly once: closed and
        # consistently oriented.
        assert all(edges.get((b, a), 0) == n for (a, b), n in edges.items()), name
        a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        assert np.einsum('ij,ij->i', a, np.cross(b, c)).sum() > 0, name     # outward


def _in_loft(v: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Whether each point lies in a loft (bottom ring, then matching top ring)."""
    n = len(v) // 2
    z0, z1 = v[:n, 2].mean(), v[n:, 2].mean()
    s = (pts[:, 2] - z0) / (z1 - z0)
    inside = np.zeros(len(pts), bool)
    for k in np.flatnonzero((s > 0) & (s < 1)):
        ring = v[:n, :2] + s[k] * (v[n:, :2] - v[:n, :2])
        inside[k] = pg.points_inside(ring, pts[k:k + 1, :2])[0]
    return inside


@pytest.mark.parametrize('case', ['round', 'hole_only'])
def test_chamfered_plate_pieces_cover_every_point_exactly_once(case):
    kw = {'plate_thickness': 0.01, **CHAMFERED[case]}
    xml = mjfit.peg(**kw, include_peg=False, center=False)
    pieces = [v for k, v in inline_meshes(xml).items() if k.startswith('plate_')]
    hole = pg.ensure_ccw(np.asarray(kw['hole'], float))
    t, h = kw['plate_thickness'], kw['hole_chamfer']
    lo = np.vstack(pieces).min(axis=0)
    hi = np.vstack(pieces).max(axis=0)
    pts = np.random.default_rng(0).uniform(lo, hi, size=(4000, 3))
    count = sum(_in_loft(v, pts).astype(int) for v in pieces)
    # The plate body sits at z = t/2, so its top is at t/2 in the meshes' frame.
    grow = np.clip(pts[:, 2] - (t / 2 - h), 0, None)
    in_hole = np.array([pg.points_inside(pg.offset_outward(hole, g), p[None, :2])[0]
                        for g, p in zip(grow, pts)])
    np.testing.assert_array_equal(count, (~in_hole).astype(int))
