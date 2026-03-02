# Narrowphase Analytical Specializations

## Overview

The narrowphase collision detection in Genesis can use general-purpose algorithms
(GJK/MPR) or hand-coded **analytical specializations** for primitive pairs.
Analytical specializations are faster and produce exact results for simple
geometry, but are tricky to get right for cylinders.

### Files

| File | Purpose |
|------|---------|
| `genesis/engine/solvers/rigid/collider/narrowphase.py` | Main narrowphase kernel; dispatches to specializations or GJK/MPR |
| `genesis/engine/solvers/rigid/collider/cylinder_contact.py` | Sphere-sphere, cylinder-sphere, cylinder-cylinder analytical functions |
| `genesis/engine/solvers/rigid/collider/capsule_contact.py` | Capsule-capsule and sphere-capsule analytical functions |
| `genesis/engine/solvers/rigid/collider/box_contact.py` | Box-box and plane-box |
| `tests/test_rigid_physics_analytical_vs_gjk.py` | Fuzz tests comparing analytical vs GJK results |

### Dispatch logic (narrowphase.py, ~line 643)

```
if sphere+sphere       → cylinder_contact.func_sphere_sphere_contact
if cylinder+cylinder   → cylinder_contact.func_cylinder_cylinder_contact
if cylinder+sphere     → cylinder_contact.func_cylinder_sphere_contact
if capsule+capsule     → capsule_contact.func_capsule_capsule_contact
if sphere+capsule      → capsule_contact.func_sphere_capsule_contact
else                   → GJK / MPR (general convex)
```

The analytical path produces a **single contact**. The narrowphase then runs a
perturbation loop using GJK to find additional multi-contact points.

### Return convention

All analytical functions return `(is_col, normal, contact_pos, penetration)`.
The **normal points from geom B to geom A**.

---

## What is a "contact"?

### Non-overlapping bodies

For two bodies that are close but not overlapping, the contact is the pair of
**witness points** — the closest point on each body's surface to the other.
The normal is the direction between them, penetration is zero (or negative =
gap distance). Well-defined and unique (for convex bodies).

### Overlapping bodies: minimum penetration depth (MPD)

For overlapping bodies, "closest surface point" is meaningless — infinitely
many points in the overlap volume are at distance zero from both surfaces.

Instead, we use the **minimum penetration depth**: the shortest translation
that would separate the two bodies. This gives:

- **Normal**: the direction of that shortest translation
- **Penetration**: the magnitude of that translation
- **Contact point**: midpoint between the two surface points along the normal

For general convex bodies, GJK detects overlap and EPA (Expanding Polytope
Algorithm) computes the MPD. For analytical primitives, we compute the MPD by
checking candidate separation directions and picking the one with **minimum**
penetration.

### Why minimum, not maximum

Multiple separation directions may be valid. The physically correct one is the
direction requiring the **least movement** to separate the bodies:

- **Barrel-barrel** direction might give pen = 0.13
- **Cap-barrel** direction might give pen = 0.017

The cap-barrel separation (pen = 0.017) is correct because pushing 0.017
along the cap normal fully separates the bodies. The barrel-barrel separation
(pen = 0.13) would also work but requires 8x more movement — it's valid but
not minimal.

Picking the maximum penetration is wrong: it gives a non-minimal separation
that wastes solver effort and produces the wrong contact normal.

### How each primitive determines the MPD

| Primitive pair | Candidate directions | Notes |
|----------------|---------------------|-------|
| **Sphere-sphere** | Line between centers (only 1) | Always gives MPD trivially |
| **Capsule-capsule** | Radial between closest axis points (only 1) | Constant-radius property guarantees this is optimal |
| **Cylinder-cylinder** | Radial between axes, A's cap normal, B's cap normal | Must check all valid candidates, pick minimum |

For capsules, there is only one candidate direction (the radial), so the
formula is trivial. For cylinders, there are multiple candidates because the
flat caps introduce additional separation directions that may require less
movement than the radial.

---

## Capsule-capsule collision (reference model)

Capsule-capsule is the simplest collision and serves as the mental model for
understanding why cylinder-cylinder is harder.

A capsule's surface is every point at distance R from its axis segment.
Hemispheres at the endpoints maintain this constant-distance property.

### Algorithm

```
Pa, Pb = closest_points_on_segments(A1, A2, B1, B2)
dist   = |Pa - Pb|
normal = (Pa - Pb) / dist
pen    = R_a + R_b - dist
surf_a = Pa - R_a * normal       # closest point on A's surface
surf_b = Pb + R_b * normal       # closest point on B's surface
contact = midpoint(surf_a, surf_b)
```

### Why it works for all cases including deep penetration

- The surface is a constant-distance offset from the axis (Minkowski sum of
  segment + sphere).
- Therefore axis distance directly equals surface distance minus 2R.
- No case splitting needed: barrel region and hemisphere cap region use the
  same formula.
- For deep penetration, `surf_a` and `surf_b` end up "inside" the other body,
  but the midpoint and normal are still geometrically consistent.
- Only degenerate case: `dist ≈ 0` (axes intersect), where normal is undefined.
  Handled via cross-product fallback.

---

## Cylinder-cylinder collision

A cylinder has three surface regions: the **barrel** (curved side) and two
**flat caps** (discs at each end). The flat caps break the constant-distance
property that makes capsule-capsule trivial.

### Why the capsule formula fails for cylinders

The barrel-barrel formula (`pen = R_a + R_b - axis_dist`) is identical to the
capsule formula. It works when both closest axis points are on the barrel
region. It fails when one or both closest points are near a cap because:

1. **False positives**: The formula assumes a hemisphere at the endpoint. A
   capsule's hemisphere extends R beyond the endpoint, but a cylinder's flat
   cap does not. Two cylinders with a gap between cap and barrel can be
   reported as colliding when they are not.

2. **Wrong penetration**: When the formula detects a real collision near a cap,
   it computes `pen = R_a + R_b - axis_dist`, which is the *capsule*
   penetration. The actual *cylinder* penetration (through the flat cap) is
   typically much smaller.

3. **Wrong normal**: The formula uses the radial direction between axis closest
   points as the normal. For a cap contact, the correct normal is the **cap
   plane normal** (the cylinder's axis direction), not the radial.

4. **Wrong contact position**: `surf_b = Pb + R_b * normal` projects along the
   radial from the axis endpoint, placing the surface point where the
   hemisphere would be. For a flat cap, this point is **outside the cylinder
   volume** entirely.

### Detecting false positives from the capsule formula

After computing the capsule-formula contact, project `surf_a` onto B's axis.
If the projection falls outside B's axis segment (on the outward side of a
cap), then `surf_a` is in the region where a capsule would have a hemisphere
but a cylinder has only empty space. The collision is a false positive.

### Contact modes for cylinder-cylinder

Given two non-parallel cylinders A and B, the contact can be:

| Mode | When | Normal | Penetration |
|------|------|--------|-------------|
| **Barrel-barrel** | Both axis closest points are interior | Radial between closest axis points | `R_a + R_b - axis_dist` |
| **Cap-barrel** | One cylinder's cap plane is crossed by the other's barrel | Cap outward normal (± axis direction) | Signed distance from barrel point to cap plane |
| **Cap-cap** | Both closest points at endpoints, axes nearly parallel | Between the two cap normals | Overlap of the two discs |
| **Edge-barrel** | Barrel near a cap rim but outside the disc | Complex (rim tangent) | Distance from barrel to rim circle |

### Cap-barrel contact geometry

The cap is a flat disc: center at an endpoint, normal = ± axis, radius = R.

To find the barrel point closest to the cap plane:

```
a       = barrel_axis · cap_normal
n_perp  = cap_normal - a * barrel_axis       # cap_normal component ⊥ to barrel axis
d_opt   = -n_perp / |n_perp|                 # barrel radial direction toward cap
t_opt   = argmin_t (signed_dist to cap)       # axial position on barrel
barrel_pt = barrel_pos + t_opt * barrel_axis + barrel_radius * d_opt
s       = (barrel_pt - cap_center) · cap_normal   # signed distance
```

If `s < 0`, the barrel penetrates the cap plane. Check that the projection of
`barrel_pt` onto the cap plane falls within the cap disc (distance from cap
center < cap_radius). If so: collision with `pen = -s`, normal = cap_normal.

### Current implementation status (WIP)

The current `func_cylinder_cylinder_contact` in `cylinder_contact.py` has:

1. **Parallel case**: Handles overlapping parallel cylinders using axis
   projection and perpendicular distance. Works correctly for barrel-barrel.
   Has the same cap issue for non-overlapping parallel segments.

2. **Non-parallel case**: Uses `closest_points_on_segments` to find Pa, Pb.
   - If either is at a segment endpoint → tries `_cap_vs_barrel` for the
     relevant cap.
   - If both are interior → barrel-barrel formula.
   - Fallback to barrel-barrel if cap-barrel finds nothing.

### Known limitations / open bugs

- **Endpoint detection is insufficient**: A cap contact can occur even when
  both closest axis points are interior to their segments. The barrel of one
  cylinder can cross the cap plane of the other while the axis closest points
  are far from the endpoints. The current code misses these cases and falls
  through to barrel-barrel, producing wrong normals and penetration.

- **Boundary case (s = 0)**: When the barrel surface is exactly at the cap
  plane, the signed distance is zero. The cap-barrel check must use a small
  margin rather than strict `s < 0` to avoid falling through to barrel-barrel.
  Currently uses `s < barrel_radius * 0.02` as margin.

- **Deep penetration**: For deeply overlapping cylinders, the barrel-barrel
  formula (capsule formula) gives large wrong penetrations. The cap-barrel
  formula gives correct cap-plane penetration, but picking the deepest among
  multiple cap candidates can select the wrong cap (the far side, giving an
  inverted normal). Must only check the cap at the clamped endpoint, not all
  caps.

- **Cap-cap contacts**: Not explicitly handled for non-parallel cylinders.
  Falls through to barrel-barrel fallback. For parallel cylinders with
  non-overlapping segments, the parallel branch uses closest-points which has
  the same capsule-formula issues.

- **Edge contacts**: When the barrel point projects outside the cap disc, the
  contact involves the cap rim (a circle). Not currently handled; falls back
  to barrel-barrel.

### Possible improved approach

Instead of endpoint detection, use `_closest_point_on_cylinder` (which already
handles barrel + flat caps correctly for single points) to find the closest
surface points:

```
surf_b = _closest_point_on_cylinder(B, Pa)   # closest point on B's surface to Pa
surf_a = _closest_point_on_cylinder(A, Pb)   # closest point on A's surface to Pb
# Iterate to refine:
surf_b = _closest_point_on_cylinder(B, surf_a)
surf_a = _closest_point_on_cylinder(A, surf_b)
```

This naturally handles all contact modes (barrel, cap, edge) without explicit
case switching. The `is_on_cap` return value from `_closest_point_on_cylinder`
indicates which surface feature is involved.

---

## Cylinder-sphere collision

Uses `_closest_point_on_cylinder` to find the closest point on the cylinder
surface to the sphere center. Then treats it like a sphere-point distance
check. The `is_inside` flag handles the case where the sphere center is inside
the cylinder volume (flips normal).

A previous bug was found here: when the sphere center penetrates the cylinder
barrel, the normal was inverted. Fixed by checking `is_inside` and flipping.

---

## Sphere-sphere collision

Trivial: `dist = |centers|`, `pen = R_a + R_b - dist`, `normal = diff/dist`.

---

## Testing

### Test file: `tests/test_rigid_physics_analytical_vs_gjk.py`

Two fuzzing approaches:

1. **Arena bounce fuzz** (`test_*_fuzz`): Multiple bodies in a walled arena
   with random bounce forces. Runs N steps with GJK, records contacts. Replays
   same qpos with analytical kernel, compares contacts per-pair per-step.
   Tests position, normal, and penetration agreement.

2. **Placement fuzz** (`test_*_placement_fuzz`): Two bodies randomly
   repositioned each step. Takes one step, compares GJK vs analytical contacts.
   Tests different configurations than arena (more random orientations, can
   produce deep penetrations).

### Tolerances

- `POS_TOL` for position and penetration errors (arena tests)
- `pos_pen_tol=0.15` for placement fuzz (looser due to deep penetrations)
- `normal_tol=-0.1` for placement fuzz (only catches inverted normals, not
  minor direction differences, because deep penetrations legitimately produce
  different contact modes between analytical and mesh-based GJK)

### ERRNO bits

The narrowphase kernel sets errno bits to track which specializations were
called. Tests verify that the analytical path was actually exercised, not
bypassed by GJK. The bits are defined in the test file as
`ANALYTICAL_ERRNO_BITS`.

---

## Key lessons learned

1. **Capsule = segment + sphere** is the gold standard for simplicity. Every
   surface point is at constant distance R from the axis. One formula, no
   cases.

2. **Cylinder flat caps break the constant-distance property.** The barrel is
   at distance R from the axis, but the cap is at distance HL along the axis.
   These are fundamentally different distance metrics.

3. **The barrel-barrel formula IS the capsule formula.** It works when both
   contact points are on barrels. It fails at caps because it assumes a
   hemisphere that doesn't exist.

4. **To detect capsule-formula false positives**: project `surf_a` onto B's
   axis. If it falls outside B's segment, the collision involves B's
   non-existent hemisphere → false positive for cylinders.

5. **Cap contact normal is the cap plane normal** (± axis direction), not the
   radial between axis closest points.

6. **For deep penetrations, only check the cap at the clamped endpoint.**
   Checking all caps and picking deepest selects the far-side cap with an
   inverted normal.

7. **Pick the MINIMUM penetration, not the maximum.** When multiple separation
   directions are valid (barrel-barrel, cap-barrel), the correct contact is
   the one with the smallest penetration — the minimum penetration depth
   (MPD). This is the shortest translation that separates the bodies. Picking
   the maximum gives a non-minimal separation with the wrong normal.

8. **Simulation divergence amplifies small errors.** Even a tiny normal
   inversion or wrong penetration in one frame causes the solver to apply wrong
   forces, which compounds over subsequent frames. The `go2` benchmark
   regression was caused by a single inverted normal in cylinder-sphere.
