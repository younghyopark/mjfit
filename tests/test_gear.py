"""Spur and internal gears: exact convex pieces and meshing at the tooth ratio."""

import mujoco
import numpy as np
import pytest
from conftest import hull_volume, inline_meshes, prism_volume

import mjfit
from mjfit import _gear
from mjfit import _polygon as pg

PAIRS = [
    dict(teeth=(18, 24)),
    dict(teeth=(12, 36), module=0.0025, tooth_width=0.46),
    dict(teeth=(32, 16), module=0.0025),
    dict(teeth=(32, 48), module=0.0016, face_width=0.005),
    dict(
        teeth=(10, 20),
        module=0.0035,
        pressure_angle=25,
        addendum=0.95,
        dedendum=1.15,
        tooth_width=0.44,
    ),
]


@pytest.mark.parametrize('kw', PAIRS, ids=lambda kw: f"{kw['teeth'][0]}-{kw['teeth'][1]}")
def test_pairs_mesh_at_the_tooth_ratio(kw):
    m = mujoco.MjModel.from_xml_string(mjfit.gear(**kw))
    d = mujoco.MjData(m)
    d.ctrl[:] = 1.0
    peak = 0.0
    for _ in range(1500):
        mujoco.mj_step(m, d)
        peak = max(peak, abs(d.qvel[1]))
    a, b = (d.qpos[m.joint(f'gear_{c}_hinge').qposadr[0]] for c in 'ab')
    assert a > np.pi  # it turned
    assert b / a == pytest.approx(-kw['teeth'][0] / kw['teeth'][1], rel=5e-3)
    assert peak < 50  # no blow-up


def test_collision_pieces_are_exact_and_tile_the_outline():
    xml = mjfit.gear(teeth=24)
    m = mujoco.MjModel.from_xml_string(xml)
    meshes = {k: v for k, v in inline_meshes(xml).items() if '_col_' in k}
    assert len(meshes) == 25  # 24 teeth and the hub
    for name, v in meshes.items():
        assert hull_volume(m, m.mesh(name).id) == pytest.approx(prism_volume(v), rel=1e-6)
    dims = _gear._dimensions(24, 0.003, np.radians(20), 1.0, 1.25, 0.48)
    outline = _gear._outline(dims, 0.0)
    x, y = outline[:, 0], outline[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert sum(prism_volume(v) for v in meshes.values()) == pytest.approx(area * 0.008, rel=1e-9)


def test_standalone_uses_pgs():
    assert 'solver="PGS"' in mjfit.gear()
    assert '<option' not in mjfit.gear(standalone=False)


@pytest.mark.parametrize(
    'kw,match',
    [
        (dict(teeth=2), 'at least 3'),
        (dict(teeth=(18, 24, 30)), 'one or two'),
        (dict(pressure_angle=50), 'between 0 and 45'),
        (dict(center_distance=0.03), 'too small'),
    ],
)
def test_invalid_input_is_rejected(kw, match):
    with pytest.raises(ValueError, match=match):
        mjfit.gear(**kw)


def test_mesh_export(tmp_path):
    path = tmp_path / 'gears.xml'
    path.write_text(mjfit.gear(mesh_dir=tmp_path / 'gears_meshes'))
    m = mujoco.MjModel.from_xml_path(str(path))
    assert m.body('gear_a').mass[0] == pytest.approx(
        mujoco.MjModel.from_xml_string(mjfit.gear()).body('gear_a').mass[0], rel=1e-6
    )


INTERNAL_PAIRS = [
    dict(teeth=(18, 54)),
    dict(teeth=(12, 36), module=0.0025, tooth_width=0.46),
    dict(teeth=(16, 48), module=0.0025),
]


@pytest.mark.parametrize(
    'kw', INTERNAL_PAIRS, ids=lambda kw: f"int-{kw['teeth'][0]}-{kw['teeth'][1]}"
)
def test_internal_pairs_mesh_at_the_tooth_ratio(kw):
    m = mujoco.MjModel.from_xml_string(mjfit.gear(internal=True, **kw))
    d = mujoco.MjData(m)
    d.ctrl[:] = 1.0
    peak = 0.0
    for _ in range(1800):
        mujoco.mj_step(m, d)
        peak = max(peak, abs(d.qvel).max())
    a, b = (d.qpos[m.joint(f'gear_{c}_hinge').qposadr[0]] for c in 'ab')
    assert a > np.pi
    # Internal pairs turn the same way; the ratio is the tooth counts.
    assert b / a == pytest.approx(kw['teeth'][0] / kw['teeth'][1], rel=5e-3)
    assert peak < 50


def test_internal_ring_pieces_are_exact_and_tile_the_rim():
    xml = mjfit.gear(teeth=40, internal=True, motor=False)
    m = mujoco.MjModel.from_xml_string(xml)
    meshes = {k: v for k, v in inline_meshes(xml).items() if '_col_' in k}
    assert meshes
    for name, v in meshes.items():
        assert hull_volume(m, m.mesh(name).id) == pytest.approx(prism_volume(v), rel=1e-6), name
    dims = _gear._dimensions(40, 0.003, np.radians(20), 1.0, 1.25, 0.48, internal=True)
    hole = _gear._internal_hole_outline(dims, 0.0)
    pieces = _gear._internal_pieces(dims, 0.0)
    assert pg.is_simple(hole) and pg.signed_area(hole) > 0
    assert all(pg.is_convex(p) for p in pieces)
    assert np.linalg.norm(hole, axis=1).max() < dims['rim_radius']
    face_width = 0.008
    assert sum(prism_volume(v) for v in meshes.values()) == pytest.approx(
        sum(pg.signed_area(p) for p in pieces) * face_width, rel=1e-9
    )


def test_internal_single_ring_has_no_center_hub():
    xml = mjfit.gear(teeth=36, internal=True, motor=False)
    assert 'gear_a_hub' not in xml
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.body('gear_a').mass[0] > 0


@pytest.mark.parametrize(
    'kw,match',
    [
        (dict(teeth=(24, 18), internal=True), 'more teeth'),
        (dict(teeth=(18, 54), internal=True, center_distance=0.08), 'too large'),
        (dict(teeth=24, internal=True, rim=0), 'rim must be positive'),
    ],
)
def test_invalid_internal_input_is_rejected(kw, match):
    with pytest.raises(ValueError, match=match):
        mjfit.gear(**kw)
