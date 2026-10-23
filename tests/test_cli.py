"""The mjfit command."""

import subprocess
import sys

import mujoco
import numpy as np
import pytest
import yaml

from mjfit.cli import main


def run(*args):
    main([str(a) for a in args])


def test_peg_shape(tmp_path):
    run('peg', '--shape', 'star', '--size', 0.012, '-o', tmp_path / 'star.xml')
    mujoco.MjModel.from_xml_path(str(tmp_path / 'star.xml'))


def test_flags_override_a_config(tmp_path, capsys):
    run('peg', '--config', 'usb_a', '--clearance', 1.5e-4, '--no-floor', '-o', tmp_path / 'a.xml')
    xml = (tmp_path / 'a.xml').read_text()
    assert 'name="floor"' not in xml
    run('peg', '--config', 'usb_a', '-o', tmp_path / 'b.xml')
    assert xml != (tmp_path / 'b.xml').read_text()


def test_outline_from_csv_and_yaml(tmp_path):
    np.savetxt(
        tmp_path / 'hole.csv',
        [[-0.005, -0.003], [0.005, -0.003], [0.005, 0.003], [-0.005, 0.003]],
        delimiter=',',
    )
    run(
        'peg',
        '--hole',
        tmp_path / 'hole.csv',
        '--island',
        '{type: circle, radius: 0.001}',
        '--plate-thickness',
        0.004,
        '--peg-length',
        0.01,
        '-o',
        tmp_path / 'p.xml',
    )
    m = mujoco.MjModel.from_xml_path(str(tmp_path / 'p.xml'))
    assert any(m.geom(i).name.startswith('island_') for i in range(m.ngeom))


def test_thread_gear_and_export(tmp_path):
    run('thread', '--config', 'm6', '--driven', 'nut', '-o', tmp_path / 'm6.xml')
    run('gear', '--teeth', 12, 30, '--export-meshes', '-o', tmp_path / 'g.xml')
    run('gear', '--teeth', 18, '--rack-teeth', 24, '-o', tmp_path / 'rack.xml')
    for f in ('m6.xml', 'g.xml', 'rack.xml'):
        mujoco.MjModel.from_xml_path(str(tmp_path / f))
    assert (tmp_path / 'g_meshes' / 'gear_a_visual.obj').exists()
    rack = mujoco.MjModel.from_xml_path(str(tmp_path / 'rack.xml'))
    assert rack.joint('rack_slide').id >= 0


def test_sample_writes_configs_models_and_grid(tmp_path):
    run(
        'sample', '-n', 3, '--family', 'convex', '--set', 'clearance=2e-4', '--grid', '-o', tmp_path
    )
    cfg = yaml.safe_load((tmp_path / 'peg_0002.yaml').read_text())
    assert cfg['clearance'] == 2e-4
    mujoco.MjModel.from_xml_path(str(tmp_path / 'grid.xml'))


def test_errors_exit_cleanly(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        run('peg', '--config', 'm10')
    assert e.value.code == 2 and 'thread config' in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run('peg', '--shape', 'star', '--clearance', 0.02)
    assert 'too large' in capsys.readouterr().err


def test_configs_listing_and_module_entry_point():
    out = subprocess.run(
        [sys.executable, '-m', 'mjfit', 'configs'], capture_output=True, text=True, check=True
    ).stdout
    assert out.splitlines()[0].startswith('peg: ') and 'm10' in out
    assert 'rack_18' in out
