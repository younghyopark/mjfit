# Gears

```python
mjfit.gear(teeth=(18, 24), module=0.003, ...)
```

`mjfit.gear` returns an MJCF model of one spur gear, or of a meshing pair, each on a
hinge about +z. `help(mjfit.gear)` and `mjfit gear --help` list every parameter.

## Geometry

- **Tooth outline.** Involute flanks above the base circle, radial flanks below it, and
  arcs at the root and the tip. `module` (pitch diameter per tooth) is shared by both
  gears, as it must be for them to mesh. `pressure_angle` is in degrees; `addendum` and
  `dedendum` are in modules.
- **Backlash** comes from `tooth_width`, the tooth's thickness at the pitch circle as a
  fraction of the circular pitch. The default, 0.48, leaves a pair 4% of a pitch of
  play. `center_distance` defaults to the sum of the pitch radii.
- **Collision.** One convex prism per tooth, closed by the chord between its root
  points, plus one hub prism through every root point. Together they are exactly the
  outline. If a tooth outline is not convex (an unusual setting), the gear falls back
  to a fan of triangles around its center.
- **Bore.** `bore` cuts a round hole of that diameter through each gear's hub. Its
  polygon circumscribes the circle (the corners stick out by at most `max_error`), so
  the bore is never smaller than drawn. The hub is then cut into one wedge per bore
  edge, bounded by the outward bisectors of the edge's corners, and the teeth are
  unchanged. `bore_chamfer` bevels the bore on both faces at 45 degrees: each wedge
  gets a band at each face whose inner edge runs from the widened mouth to the bore,
  and every piece stays exactly convex.
  The angle is 45 degrees, for the reason given for pegs.
- **Rendering.** A separate visual mesh renders the outline, and the collision prisms
  sit in geom group 3, hidden by default (press 3 in MuJoCo's viewer to show them).

## The scene

The gears stand on short visual axles above a floor. With `shafts=True` (which needs a
`bore`) each gear sits instead on a free joint around a fixed collision shaft, as in a
gear assembly task: the shaft is the bore moved in by `clearance` on every edge,
standing on the floor and reaching above the gear, and `shaft_chamfer` bevels its top.
The gears rest on the floor and can be lifted off and slid back on; the bodies are
still `gear_a` and `gear_b`, with free joints `gear_a_joint` and `gear_b_joint` and
shafts `gear_a_shaft` and `gear_b_shaft`. A bored hub touches a surface with many small
pieces, so standalone files with shafts reserve a larger constraint arena
(`<size memory="256M"/>`); scenes of your own may need the same. `motor=True` adds a motor,
`drive_gear_a`, on the first gear's hinge, with control in [-1, 1] times `motor_gear`
(N m); it is on by default for hinged gears and unavailable on shafts. The bodies are `gear_a` and `gear_b`, the hinges `gear_a_hinge` and
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
- Only external spur gears; no helical, bevel, or internal gears.
- Torque passes between the gears through tooth contact alone; no gear constraint is
  added.
