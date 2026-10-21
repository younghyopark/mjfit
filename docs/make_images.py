"""Render the README's images from the bundled configs.

    MUJOCO_GL=egl python docs/make_images.py

Every collision piece gets its own color, so the convex decomposition shows.
"""

import colorsys
import os
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import mjfit

OUT = Path(__file__).resolve().parent / 'images'
FONT = '/usr/share/fonts/truetype/lato/Lato-Regular.ttf'
TILE = (720, 540)
GALLERY = [
    ('star', 'Star hole'),
    ('usb_c', 'USB-C, hollow peg'),
    ('db9', 'D-sub, nine pins'),
    ('rounded_convex', 'Rounded peg, handle, plate'),
    ('m10', 'M10 bolt and nut (cut open)'),
    ('spur_pair', '18:24 spur gears'),
]


def studio(xml_or_spec) -> mujoco.MjModel:
    """Compile with a white backdrop, an endless floor and soft shadows."""
    spec = (
        xml_or_spec
        if isinstance(xml_or_spec, mujoco.MjSpec)
        else mujoco.MjSpec.from_string(xml_or_spec)
    )
    spec.add_texture(
        name='backdrop',
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_FLAT,
        rgb1=[1, 1, 1],
        rgb2=[1, 1, 1],
        width=16,
        height=16,
    )
    model = spec.compile()
    floor = model.geom('floor').id
    model.geom_size[floor][:2] = 50.0
    model.geom_rgba[floor] = (1, 1, 1, 1)
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.45
    model.light_diffuse[:] = 0.45
    return model


def piece_colors(model: mujoco.MjModel, hues: dict[str, float], spread: float = 1.0) -> None:
    """One color per collision geom: a body's hue, varied in lightness and saturation.

    `spread` scales the variation. Hundreds of millimeter-sized pieces, as in a
    thread, need little of it, or they read as noise.
    """
    rng = np.random.default_rng(0)
    for g in range(model.ngeom):
        body = model.body(model.geom_bodyid[g]).name
        name = model.geom(g).name
        # A visual mesh covers the same surface as the pieces and would z-fight with them.
        if name.endswith('_visual') or name.endswith('_hub'):
            model.geom_rgba[g, 3] = 0.0
        if model.geom_contype[g] == 0 or body == 'world':
            continue
        hue = hues.get(body, 0.6) + spread * rng.uniform(-0.04, 0.04)
        light = 0.6 + spread * rng.uniform(-0.15, 0.15)
        sat = 0.625 + spread * rng.uniform(-0.175, 0.175)
        model.geom_rgba[g] = (*colorsys.hls_to_rgb(hue % 1, light, sat), 1.0)
        model.geom_group[g] = 0


def frame(model, data, cam, scale=1.0, azimuth=135.0, elevation=-32.0):
    """Fit everything that is drawn into a 30 degree view."""
    corners = []
    for g in range(model.ngeom):
        if model.geom_rgba[g, 3] == 0 or model.geom(g).name == 'floor':
            continue
        center, half = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        box = center + half * np.array(
            [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        )
        corners.append(box @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    corners = np.vstack(corners)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    model.vis.global_.fovy = 30.0
    cam.lookat[:] = (lo + hi) / 2
    cam.distance = 1.15 * scale * np.linalg.norm(hi - lo) / 2 / np.sin(np.radians(15))
    cam.azimuth, cam.elevation = azimuth, elevation


def render(model, data, size=TILE, **view) -> Image.Image:
    model.vis.global_.offwidth, model.vis.global_.offheight = size
    model.vis.quality.shadowsize = 4096
    renderer = mujoco.Renderer(model, size[1], size[0])
    cam = mujoco.MjvCamera()
    frame(model, data, cam, **view)
    renderer.update_scene(data, cam)
    image = Image.fromarray(renderer.render())
    renderer.close()
    return image


def tile(name: str) -> tuple[Image.Image, int]:
    kind = mjfit.configs()[name]
    model = studio(mjfit.build(name))
    data = mujoco.MjData(model)
    piece_colors(
        model,
        {'plate': 0.58, 'peg': 0.02, 'nut': 0.58, 'bolt': 0.08, 'gear_a': 0.58, 'gear_b': 0.08},
        spread=0.2 if kind == 'thread' else 1.0,
    )
    if kind == 'peg':  # lift the peg out of the hole so both show
        j = model.joint('peg_joint').qposadr[0]
        peg_len = mjfit.load(name).get('peg_length', 0.03)
        data.qpos[j + 2] += 0.9 * peg_len
    if kind == 'thread':  # screw the bolt down to the nut's base; whole leads keep the teeth meshed
        cfg = mjfit.load(name)
        lead = cfg['pitch'] * (cfg.get('n_starts') or 1)
        j = model.joint('bolt_joint').qposadr[0]
        turns = np.floor((data.qpos[j + 2] - model.body('nut').pos[2]) / lead + 1e-6)
        data.qpos[j + 2] -= turns * lead
    mujoco.mj_forward(model, data)
    if kind == 'thread':  # cut the nut open
        for g in range(model.ngeom):
            if model.body(model.geom_bodyid[g]).name == 'nut' and data.geom_xpos[g][1] < 0:
                model.geom_rgba[g, 3] = 0.0
    pieces = sum(
        1
        for g in range(model.ngeom)
        if model.geom_contype[g] and model.body(model.geom_bodyid[g]).name != 'world'
    )
    view = {'elevation': -40.0} if kind == 'gear' else {}
    return render(model, data, **view), pieces


def gallery() -> None:
    font = ImageFont.truetype(FONT, 30)
    cols, rows = 3, 2
    pad = 64
    sheet = Image.new('RGB', (cols * TILE[0], rows * (TILE[1] + pad)), 'white')
    draw = ImageDraw.Draw(sheet)
    for k, (name, label) in enumerate(GALLERY):
        image, pieces = tile(name)
        x, y = (k % cols) * TILE[0], (k // cols) * (TILE[1] + pad)
        sheet.paste(image, (x, y))
        text = f'{label}  ·  {pieces} convex pieces'
        w = draw.textlength(text, font=font)
        draw.text((x + (TILE[0] - w) / 2, y + TILE[1] + 12), text, fill=(90, 90, 90), font=font)
    sheet.save(OUT / 'gallery.png', optimize=True)


def sampled_sheet() -> None:
    """Fifteen random assemblies of the convex family, each framed alike."""
    cols, rows, size = 5, 3, (480, 400)
    sheet = Image.new('RGB', (cols * size[0], rows * size[1]), 'white')
    for k, cfg in enumerate(mjfit.sample_pegs(cols * rows, seed=0, family='convex')):
        model = studio(mjfit.build(cfg))
        data = mujoco.MjData(model)
        piece_colors(model, {'plate': 0.58, 'peg': 0.02})
        data.qpos[model.joint('peg_joint').qposadr[0] + 2] += 0.25 * cfg['peg_length']
        mujoco.mj_forward(model, data)
        sheet.paste(
            render(model, data, size=size, scale=0.9, elevation=-35.0),
            ((k % cols) * size[0], (k // cols) * size[1]),
        )
    sheet.save(OUT / 'sampled.png', optimize=True)


if __name__ == '__main__':
    os.environ.setdefault('MUJOCO_GL', 'egl')
    OUT.mkdir(exist_ok=True)
    gallery()
    sampled_sheet()
    print(f'wrote {OUT}')
