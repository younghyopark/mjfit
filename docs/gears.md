# Gears

```python
mjfit.gear(teeth=(18, 24), module=0.003, ...)
mjfit.gear(teeth=18, rack_teeth=48, ...)
```

`mjfit.gear` returns an MJCF model of one spur gear, a rack and pinion, or a meshing
pair. Gears turn on a hinge about +z; a rack slides along +x. `help(mjfit.gear)` and
`mjfit gear --help` list every parameter.

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
  to a fan of triangles around its center. A rack tooth is a convex trapezoid with
  straight flanks at the pressure angle, on a rectangular backing bar.
- **Rendering.** A separate visual mesh renders the outline, and the collision prisms
  sit in geom group 3, hidden by default (press 3 in MuJoCo's viewer to show them).
- **Rack and pinion.** `rack_teeth` builds a linear rack with that many teeth and a
  pinion of `teeth` (a single count). The rack body is `rack`, on slide joint
  `rack_slide` along +x. The pinion sits one pitch radius above the rack pitch line
  (or `center_distance` above it). They roll at the pitch radius: rack travel equals
  pinion angle times that radius. `rack_backing` is the bar thickness below the
  roots, in modules. The rack is finite; past the last tooth the pinion freewheels.

## The scene

The gears stand on short visual axles above a floor. `motor=True` adds a motor,
`drive_gear_a`, on the first gear's hinge, with control in [-1, 1] times `motor_gear`
(N m). The bodies are `gear_a` and `gear_b`, the hinges `gear_a_hinge` and
`gear_b_hinge`. A rack-and-pinion scene uses `gear_a` and `rack`.

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
- External spur gears and a straight rack; no helical, bevel, or internal gears.
- Torque passes between the gears through tooth contact alone; no gear constraint is
  added.
