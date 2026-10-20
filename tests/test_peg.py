"""Peg-in-hole geometry: exact convex pieces, clearance, frames and masses."""

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
