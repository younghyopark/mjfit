# Gears

```python
mjfit.gear(teeth=(18, 24), module=0.003, ...)
mjfit.gear(teeth=(18, 54), internal=True, ...)
```

`mjfit.gear` returns an MJCF model of one spur gear, a ring gear, or a meshing pair,
each on a hinge about +z. `help(mjfit.gear)` and `mjfit gear --help` list every
parameter.

## Geometry

- **Tooth outline.** Involute flanks above the base circle, radial flanks below it, and
  arcs at the root and the tip. `module` (pitch diameter per tooth) is shared by both
  gears, as it must be for them to mesh. `pressure_angle` is in degrees; `addendum` and
  `dedendum` are in modules.
- **Backlash** comes from `tooth_width`, the tooth's thickness at the pitch circle as a
  fraction of the circular pitch. The default, 0.48, leaves a pair 4% of a pitch of
  play. `center_distance` defaults to the sum of the pitch radii, or to their
  difference when `internal=True`.
- **Collision.** One convex prism per tooth, closed by the chord between its root
  points, plus one hub prism through every root point. Together they are exactly the
  outline. If a tooth outline is not convex (an unusual setting), the gear falls back
  to a fan of triangles around its center. A ring gear is convex wedges from each
  inner-edge out to the rim.
- **Rendering.** A separate visual mesh renders the outline, and the collision prisms
  sit in geom group 3, hidden by default (press 3 in MuJoCo's viewer to show them).
- **Internal gears.** `internal=True` puts teeth on the inside of a ring. One tooth
  count builds a ring alone; two counts put an external pinion inside the ring (the
  second count is the ring and must be larger). They turn the same way, at the ratio
  of their tooth counts. `rim` is the radial thickness outside the ring's roots, in
  modules.

## The scene

The gears stand on short visual axles above a floor. `motor=True` adds a motor,
`drive_gear_a`, on the first gear's hinge, with control in [-1, 1] times `motor_gear`
(N m). The bodies are `gear_a` and `gear_b`, the hinges `gear_a_hinge` and
`gear_b_hinge`.

## Use PGS for hinged gears

A gear on a hinge through its center of mass cannot translate, so MuJoCo's diagonal
approximation gives its contacts almost no regularization. The default Newton solver
then fails: in our tests no stock pair meshed at its ratio, most drove the second gear
the wrong way, and some spun it at thousands of radians per second. The PGS solver
meshes them at the exact ratio, within half a percent, so standalone gear files use
`solver="PGS"`. Keep PGS for hinged gears in your own scenes. A gear on a free joint,
held by a gripper as in the paper's gear meshing task, is not affected.

## Limitations

- The tooth outline is an involute approximation with an unfilleted root, not a
  manufacturing-grade CAD gear.
- External spur gears and internal ring gears; no helical or bevel gears.
- Torque passes between the gears through tooth contact alone; no gear constraint is
  added.
