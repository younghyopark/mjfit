"""Spur gears: exact convex pieces and meshing at the tooth ratio."""

import mujoco
import numpy as np
import pytest
from conftest import hull_volume, inline_meshes, prism_volume

import mjfit
from mjfit import _gear

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


# ----- bores and shafts -------------------------------------------------------

from mjfit import _polygon as pg  # noqa: E402
from mjfit import shapes  # noqa: E402

BORED = dict(teeth=24, module=0.002, face_width=0.01, bore=0.0105, bore_chamfer=7e-4)


def _loft_volume(v: np.ndarray) -> float:
    n = len(v) // 2
    lo, hi = v[:n, :2], v[n:, :2]
    area = [abs(pg.signed_area(r)) for r in (lo, (lo + hi) / 2, hi)]
    return (v[n:, 2].mean() - v[:n, 2].mean()) / 6 * (area[0] + 4 * area[1] + area[2])


@pytest.mark.parametrize('chamfer', [0.0, 7e-4])
def test_bored_hub_pieces_are_exact_and_have_the_bored_volume(chamfer):
    xml = mjfit.gear(**{**BORED, 'bore_chamfer': chamfer})
    m = mujoco.MjModel.from_xml_string(xml)
    meshes = {k: v for k, v in inline_meshes(xml).items() if '_col_' in k}
    for name, v in meshes.items():
        assert hull_volume(m, m.mesh(name).id) == pytest.approx(_loft_volume(v), rel=1e-6), name
    dims = _gear._dimensions(24, 0.002, np.radians(20), 1.0, 1.25, 0.48)
    gear = abs(pg.signed_area(_gear._outline(dims, 0.0))) * 0.01
    bore = shapes.circle(0.00525, max_error=1e-5)
    mouth = pg.offset_outward(bore, chamfer)
    band = _loft_volume(np.vstack([np.column_stack([bore, np.zeros(len(bore))]),
                                   np.column_stack([mouth, np.full(len(bore), chamfer)])]))
    hole = abs(pg.signed_area(bore)) * (0.01 - 2 * chamfer) + 2 * band
    assert sum(_loft_volume(v) for v in meshes.values()) == pytest.approx(gear - hole, rel=1e-9)


def test_bore_is_never_smaller_than_drawn_and_the_shaft_keeps_the_clearance():
    c = 2e-4
    xml = mjfit.gear(**BORED, shafts=True, clearance=c, shaft_chamfer=1e-3)
    meshes = inline_meshes(xml)
    shaft = meshes['gear_a_shaft'][:, :2]
    ring = np.vstack([v[:, :2] for k, v in meshes.items() if '_col_' in k])
    r_bore = np.linalg.norm(ring, axis=1).min()
    assert 0.00525 <= r_bore <= 0.00525 + 1e-5                         # corners outside, by max_error
    r_shaft = np.linalg.norm(shaft, axis=1)
    assert r_shaft.max() < 0.00525 - c + 1e-5                          # corners inside the circle
    edges = np.roll(shaft[: len(shaft) // 2], -1, axis=0) - shaft[: len(shaft) // 2]
    mids = shaft[: len(shaft) // 2] + edges / 2
    np.testing.assert_allclose(np.linalg.norm(mids, axis=1), 0.00525 - c, rtol=1e-9)
    tip = meshes['gear_a_shaft_tip']
    n = len(tip) // 2
    np.testing.assert_allclose(tip[n:, :2], pg.offset_inward(tip[:n, :2], 1e-3), atol=1e-15)


def test_gears_on_shafts_are_free_and_slide_down_to_the_floor():
    m = mujoco.MjModel.from_xml_string(mjfit.gear(**BORED, shafts=True, clearance=5e-4,
                                                  shaft_chamfer=1e-3))
    assert m.nu == 0 and m.joint('gear_a_joint').type == mujoco.mjtJoint.mjJNT_FREE
    d = mujoco.MjData(m)
    q = m.joint('gear_a_joint').qposadr[0]
    start = d.qpos[q + 2]
    mujoco.mj_step(m, d, nstep=200)
    assert d.qpos[q + 2] < start                                        # it fell...
    assert d.qpos[q + 2] == pytest.approx(0.005, abs=2e-4)              # ...onto the floor
    assert np.hypot(*d.qpos[q:q + 2]) < 5e-4 + 1e-5                     # around its shaft


@pytest.mark.parametrize('kw,match', [
    (dict(bore_chamfer=1e-3), 'need a bore'),
    (dict(shafts=True), 'need a bore'),
    (dict(bore=0.01, shaft_chamfer=1e-3), 'needs shafts'),
    (dict(bore=0.01, shafts=True, motor=True), 'motor'),
    (dict(bore=0.01, bore_chamfer=0.004), 'half the face_width'),
    (dict(bore=0.07), 'root circle'),
    (dict(bore=0.01, shafts=True, clearance=0.006), 'too large'),
    (dict(bore=0.01, shafts=True, clearance=1e-5, max_error=1e-4), 'max_error'),
    (dict(bore=-0.01), 'nonnegative'),
])
def test_invalid_bores_are_rejected(kw, match):
    with pytest.raises(ValueError, match=match):
        mjfit.gear(teeth=24, **kw)


def test_bored_mesh_export_matches(tmp_path):
    kw = dict(**BORED, shafts=True, shaft_chamfer=1e-3, clearance=5e-4)
    path = tmp_path / 'bored.xml'
    path.write_text(mjfit.gear(**kw, mesh_dir=tmp_path / 'meshes'))
    exported = mujoco.MjModel.from_xml_path(str(path))
    inline = mujoco.MjModel.from_xml_string(mjfit.gear(**kw))
    assert exported.body('gear_a').mass[0] == pytest.approx(inline.body('gear_a').mass[0], rel=1e-6)
