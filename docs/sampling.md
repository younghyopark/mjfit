# Random assemblies

```python
configs = mjfit.sample_pegs(1000, seed=0, family='convex')
xmls = [mjfit.build(cfg) for cfg in configs]
```

`mjfit.sample_pegs(n, seed, family, **options)` returns `n` peg configs.
`mjfit.sample_peg(seed, index, family, **options)` returns one. A config is a plain
dict, so it can be saved as YAML, inspected, edited, and built with `mjfit.build`.
Sample `index` of `seed` comes from its own random stream, so it is the same however
many samples you draw. From the command line, `mjfit sample -n 100 --family convex
-o samples/` writes each sample's YAML and MJCF, and `--grid` adds all of them in one
`grid.xml`.

Every option takes a fixed value or a `(low, high)` range to draw from uniformly:

```python
mjfit.sample_pegs(100, clearance=(2e-4, 5e-4), islands=2, handle_probability=0.0)
```

The `primitive` family draws its round holes and islands to the default `max_error` of
0.1 mm, so give it clearances well above that ([Curves](pegs.md#outlines)); below it,
`mjfit.peg` warns.

## The `convex` family

Random convex outlines with exactly `vertices` vertices (drawn with Valtr's algorithm),
spanning `width` by `width / aspect`. Their corners are styled at random by the
`rounding` probabilities: all sharp, a random subset rounded by `corner_rounding`, or
all fully rounded with quadratic Bezier curves. A handle (with probability
`handle_probability`) is another random rounded convex outline, scaled to contain the
peg with between `handle_min_margin` and `handle_max_margin` to spare. The plate is a
third one, containing the hole and the handle with a rim between `plate_min_rim` and
`plate_max_rim`.

| Option | Default |
| --- | --- |
| `vertices` | 3 to 10 |
| `width` | 32 to 48 mm |
| `aspect` | 20/7 |
| `rounding` | ⅓ sharp, ⅓ mixed, ⅓ smooth |
| `corner_rounding` | 0.2 to 0.5 |
| `clearance` | 0.5 mm |
| `total_height` | 50 to 100 mm, handle included |
| `min_insertion` | 22 mm under a handle |
| `handle_probability`, `handle_height` | 0.5; 20 to 40 mm |
| `handle_min_margin`, `handle_max_margin` | 1 mm, 10 mm |
| `plate_thickness` | 10 to 20 mm |
| `plate_min_rim`, `plate_max_rim` | 5 mm, 30 mm |

The outline, height, and handle defaults follow the peg family of the paper's training
distribution.

## The `primitive` family

A hole drawn from `hole_shapes` (circle, rectangle, stadium, regular polygon, star) of
size `size`, rotated at random, with up to `islands` islands drawn from `island_shapes`.
Islands are placed by rejection sampling and must leave the hollow peg walls of at
least `min_wall`, both around and between their pockets. If an island does not fit
after `attempts` tries, fewer are placed.

| Option | Default |
| --- | --- |
| `size` | 25 mm circumradius |
| `clearance` | 0.5 mm |
| `islands` | 0 to 3 |
| `plate_thickness` | 10 to 30 mm |
| `plate_scale` | plate side 5 to 10 times `size` |
| `peg_overhang` | peg 20 to 60 mm longer than the plate is thick |
| `handle_probability`, `handle_height`, `handle_margin` | 0.5; 10 to 40 mm; 2 to 6 mm |
| `min_wall` | 1 mm |

## Many assemblies in one scene

`mjfit.grid(parts, spacing)` lays MJCF models out on a square grid in one
`mujoco.MjSpec`. Every element of part `k` gets the prefix `p<k>_`, its default class
included, so copies never clash. Build the parts with `standalone=False`; the grid adds
one floor, a light, and the solver options.

```python
parts = [mjfit.build(cfg, standalone=False) for cfg in configs[:25]]
spec = mjfit.grid(parts, spacing=0.2)
model = spec.compile()
```

To give each of many parallel worlds a different assembly, as in the paper's training,
build the variants in memory this way and hand them to your simulator's mechanism for
per-world models.
