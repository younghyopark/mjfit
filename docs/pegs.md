# Pegs and holes

```python
mjfit.peg(hole=..., clearance=1e-4, ...)   # or peg=... instead of hole=...
```

`mjfit.peg` returns an MJCF model of a plate with a hole and the peg that fits it. Every
collision geom is a convex prism, so MuJoCo collides the parts natively and the
clearance between them is exact. `help(mjfit.peg)` and `mjfit peg --help` list every
parameter.

## Outlines

Give the cross-section of either the `hole` or the `peg`. The other part is its exact
offset by `clearance`: every edge moves along its normal by the clearance and the
corners are mitred, so each peg edge sits exactly `clearance` from its hole edge,
whatever the shape.

An outline is either an `(n, 2)` array of vertices, in meters, or a shape spec, the
same mapping a config file uses. `mjfit.shapes` builds both:

| Function | Shape |
| --- | --- |
| `circle(radius, n=None, max_error=1e-4)` | A regular polygon around a circle |
| `rect(width, height)` | A rectangle |
| `stadium(width, height, n=None, max_error=1e-4)` | A rectangle with round ends |
| `trapezoid(bottom_width, top_width, height)` | An isosceles trapezoid |
| `ngon(n, radius)` | A regular polygon with the given circumradius |
| `star(points, outer_radius, inner_radius)` | A star |
| `polygon(vertices)` | Any simple polygon |
| `bezier(vertices, rounding=0.5, max_error=1e-4)` | A convex polygon with rounded corners |
| `csv(path)` | A polygon read from `x,y` lines |

Every function also takes `center=(x, y)` and `rotate` (in degrees), and returns a
counterclockwise array. In a config the same shapes are mappings:
`{type: star, points: 5, outer_radius: 0.012, inner_radius: 0.0054, rotate: 18}`.
`shapes.preset(name, size)` gives the named holes of `mjfit peg --shape`.

**Curves.** A circle becomes a polygon that circumscribes it: its flats touch the circle
and its corners stick out by at most `max_error` (or use exactly `n` sides). A hole is
therefore never smaller than drawn. The peg is the hole's exact offset, so the
clearance holds along every flat and grows only at the `n` corners, by
`clearance * (1/cos(pi/n) - 1)`. Stadium ends and round islands work the same way, so
an island is never thinner than drawn. `bezier` instead samples points on its curves
(each chord cuts in by at most `max_error`): there the polygon is the shape.

The clearance is exact between the polygons, but each polygon strays from its curve
by up to `max_error`. Against the round parts they stand for, the peg's corners stick
out nearly as far as the hole's, and an island's corners reach as far into the gap a
real plug leaves around it. Keep `max_error` well below the clearance; a tenth keeps
every part within a tenth of the clearance of the real one. `mjfit.peg` warns when a
curve of the hole, the peg, or an island strays further than the clearance.

## Islands and hollow pegs

`islands` are outlines of material standing inside the hole, as tall as the plate, such
as the tongue of a USB receptacle or the pins of a D-sub. The peg then becomes a hollow
shell: a ring with one pocket per island, each pocket the island grown by the
clearance, and a solid back above the pockets. Islands must lie inside the hole and
their pockets inside the peg, apart from each other. Outlines too close for the
clearance raise an error that names them.

## Handle, plate, and bottom

- **Handle.** `handle_height > 0` adds a wider grip section on top of the peg, in the
  same body. Its outline is the `handle` you give (convex, containing the peg) or, by
  default, the peg's convex hull grown by `handle_margin`. A margin larger than the
  clearance keeps the handle from entering the hole.
- **Plate.** By default the plate is a square of side `plate_size`. A convex `plate`
  outline replaces it; `min_rim` then sets the least distance from its edges to the
  hole and the handle.
- **Bottom.** `bottom_thickness > 0` closes the hole from below with a slab following
  the plate's outline.
- **Mass.** `peg_mass` fixes the total mass of the peg and handle, spread over their
  geoms by volume. Otherwise each geom gets `peg_density` (or `plate_density`).

## How the parts are cut into convex pieces

- **Plate.** If the hole is star-shaped from its center (every convex hole, and stars
  and slots), the square plate is cut into one wedge per hole edge, bounded by
  the edge and the rays through its two ends. Otherwise, and for an explicit plate
  outline, the plate minus the hole is triangulated (with
  [earcut](https://github.com/mapbox/earcut)).
- **Peg.** A convex peg is one prism. A star-shaped one is fanned from its center, and
  anything else is ear-clipped into triangles.
- **Hollow peg.** The ring is triangulated around its pockets, and the solid back is cut
  like a solid peg.
- **Islands** are single prisms if convex, else ear-clipped.

Every piece is a counterclockwise convex polygon extruded into a prism. MuJoCo takes
each mesh's convex hull, which for these prisms is the prism itself. The tests check
this by comparing every compiled hull's volume with its prism's.

## Frames and names

The plate body `plate` stands on z = 0, raised by `bottom_thickness` if the hole has a
bottom. The peg body `peg` has a free joint `peg_joint`, and its origin is the middle of
the insertion section. It starts `spawn_height` above the plate, directly over the
hole. With `center=True` (the default) all outlines move together so that the given
outline's centroid is at the origin; `center=False` keeps your coordinates.

Geoms and meshes share names: `plate_<k>`, `island_<k>`, and `bottom` on the plate;
`peg_<k>`, `peg_ring_<k>`, `peg_back_<k>`, and `handle_<k>` on the peg.

## Contact

Each geom is in the default class `mjfit`, with `solref="0.002 1"`,
`solimp="0.99 0.999 0.001"`, `condim=4`, and `friction="1 0.01 0.001"`. The two
contact settings are parameters (`friction`, `condim`). The 2 ms time constant needs a
time step of 1 ms or less (`timestep`, used in standalone files).

## Limitations

- Outlines must be simple polygons. Very sharp convex corners make the mitred offset
  reach far: at a corner of angle `a` the offset vertex moves `clearance / sin(a/2)`.
  Offsets that would fold an outline over itself raise an error instead.
- An explicit `plate` or `handle` outline must be convex.
- Islands are as tall as the plate, and the hollow peg's pockets as deep as the plate
  plus the clearance.
