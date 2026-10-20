"""Bundled configs and the config loader."""

import warnings

import mujoco
import numpy as np
import pytest

import mjfit


@pytest.mark.parametrize('name', sorted(mjfit.configs()))
def test_bundled_config_builds_and_settles(name):
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # curves drawn well within the clearance
        model = mujoco.MjModel.from_xml_string(mjfit.build(name))
    data = mujoco.MjData(model)
    for _ in range(300):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()


@pytest.mark.parametrize('name', [n for n, k in mjfit.configs().items() if k == 'peg'])
def test_aligned_peg_passes_the_hole_without_touching_it(name):
    """Dropped centered, a peg reaches the floor (or the hole's bottom) without
    touching the walls or islands: the clearance is exact."""
    model = mujoco.MjModel.from_xml_string(mjfit.build(name))
    data = mujoco.MjData(model)
    plate, floor = model.body('plate').id, model.geom('floor').id
    bottom = (
        model.geom('bottom').id
        if 'bottom' in [model.geom(i).name for i in range(model.ngeom)]
        else -1
    )
    for _ in range(1000):
        mujoco.mj_step(model, data)
        geoms = {g for c in data.contact[: data.ncon] for g in c.geom}
        if floor in geoms or bottom in geoms:
            break
        touching = [
            c
            for c in data.contact[: data.ncon]
            if plate in (model.geom_bodyid[c.geom[0]], model.geom_bodyid[c.geom[1]])
        ]
        assert not touching, f'{name}: the peg touched the plate before landing'
    else:
        raise AssertionError(f'{name}: the peg never landed')


def test_unknown_keys_are_errors_with_suggestions():
    with pytest.raises(ValueError, match="'clearence' \\(did you mean 'clearance'\\?\\)"):
        mjfit.build({'kind': 'peg', 'hole': [[0, 0], [0.01, 0], [0, 0.01]], 'clearence': 1e-4})
    with pytest.raises(ValueError, match='kind'):
        mjfit.build({'hole': [[0, 0], [0.01, 0], [0, 0.01]]})


def test_overrides_and_paths(tmp_path):
    a = mjfit.build('usb_a', clearance=2e-4)
    assert a != mjfit.build('usb_a')
    np.savetxt(
        tmp_path / 'hole.csv',
        [[-0.005, -0.003], [0.005, -0.003], [0.005, 0.003], [-0.005, 0.003]],
        delimiter=',',
    )
    (tmp_path / 'cfg.yaml').write_text('kind: peg\nhole: {type: csv, path: hole.csv}\n')
    mujoco.MjModel.from_xml_string(mjfit.build(tmp_path / 'cfg.yaml'))
    with pytest.raises(FileNotFoundError, match='bundled configs'):
        mjfit.build('no_such_config')
