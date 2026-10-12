# Sling lift harness — notes

**Approach.** The candidate and the reference go through the same measurement stage. The candidate is then scored *relative to* the reference, never against an incidental property of the reference file.

- Parts are matched by mass properties: volume plus principal radii of gyration. These don't change under motion, reflection, renaming or reordering.
- Joints are matched by the parts they connect.
- Where a component measures "how much" (how constrained, how parametric), the candidate is normalised by the reference's level. The reference scores 1 by construction.

The harness is fully deterministic and doesn't use the LLM judge, because every question turned out to be measurable.

## What is measured

The components and weights are unchanged, and `max_score` = 6.

| Component | Measurement |
|---|---|
| executes and builds geometry (gate) | The document opens and recomputes in `freecadcmd` and holds solid volume. |
| geometry matches ground truth | The part set, **pose-free**: a Hungarian match of part mass signatures, volume-weighted, read at solid, body and welded-group level. Up to 5% difference by volume is free, because the drawing doesn't dimension every part. The score reaches 0 at 20%. |
| pantograph joints mated | The share of the reference's pivots realised by an *honest* joint between the matching parts. Honest means: active; between two different rigid parts; each frame on its part's material or on one of its hole or pin axes, with no gap-bridging offset; and coincident at the delivered pose. |
| mechanism articulates | Instantaneous mobility from the rank of the joint-constraint Jacobian, computed by the scorer itself. It is compared with the reference, allowing one spare spinning pin. It is multiplied by a limits score: limits must be well-formed, contain the delivered pose, and a slider's stroke must be able to **reach the drawn configuration**. The reach test is physical: in the reference's space, how far must the part slide, and does the stroke allow it? |
| carries dimensional constraints | Share of solid volume with modelling history (not dead imports) × how constrained the sketches that feed solids are. Both are relative to the reference. |
| dimensions drive the geometry | Up to 16 dimensions (spreadsheet cells, sketch constraints, feature lengths) are nudged ±5% and the document recomputed. The probe checks whether the **solids** changed, compared by placement-free volume and area. Decoy dimensions change nothing. |
| signature dimensions present | Of the drawing's key dimensions (687, 511, 453, 216, 228.5, 120, 100, 65, Ø16, Ø20), only those the reference itself exhibits are expected. Each must appear as a parameter, a diameter, a pin spacing or an extent. |

## What I let go

- **Names, labels, tree shape, part order, flat vs. sub-assembly structure.** Sub-assemblies are flattened, and their joints are mapped into the top-level frame.
- **The delivered pose.** For an articulating mechanism it's a free variable. What matters is whether the mechanism can reach the drawn configuration.
- **Absolute joint-limit values.** These depend on where the frames were placed, so I compare what the limits permit physically instead.
- **Constraint counts.** A count rewards decoys, so I probe whether dimensions drive the geometry instead.
- **Joint limits as a requirement.** The brief doesn't ask for them, so having none is not penalised.

## The mirror: not a defect

- Nothing the brief asks for changes under reflection: the parts, the pivots, the mobility and the driving dimensions are all unaffected.
- Misreading the projection convention on this drawing produces exactly this result.
- Penalising it would also fail an honest candidate who built the lift "the other way round".

The harness still detects and reports the reflection. Setting `MIRROR_IS_DEFECT` in `scoring.py` penalises it.

## Scores: real corpus, FreeCAD 1.1.4

| Model | Score | Geom | Mated | Artic | Cons | Drive | Sig |
|---|---|---|---|---|---|---|---|
| **reference** | **6.00** | 1 | 1 | 1 | 1 | 1 | 1 |
| assembly_mirrored | 6.00 | 1 | 1 | 1 | 1 | 1 | 1 |
| half_finished_missing_features | 5.86 | **0.86** | 1 | 1 | 1 | 1 | 1 |
| joint_limits_swapped | 5.50 | 1 | 1 | **0.50** | 1 | 1 | 1 |
| one_link_missing_joints_bridge_gap | 5.17 | 1 | **0.67** | **0.50** | 1 | 1 | 1 |
| dummy_joints_inflate_the_count | 4.33 | 1 | **0.33** | **0.00** | 1 | 1 | 1 |

Each model keeps 1.0 on every component it doesn't break.

- **half_finished:** 7.5% of the material is missing, including the master link.
- **joint_limits_swapped:** the stroke allows −50..0 mm, but reaching the drawn configuration needs +169 mm.
- **one_link_missing:** 6 of 9 pivots are honest, the bridging joint is caught off the material, and the mechanism has 6 DoF instead of 3.
- **dummy_joints:** only 3 of 9 pivots are real, and the joint set locks the mechanism.

`selftest.py` (no FreeCAD needed) also passes two honest re-modellings at 6/6. It also checks the two extra adversaries named in `task.toml`: decorative dimensions lose "drive", and a 2× scale loses geometry.

## Observations

- `task.toml`'s `notes` field contains an audit with expected scores per example, which partly gives away the intended answers.
- The examples are built from sub-assemblies, following the drawing's parts list, and include the two spring-type pins. The reference is flat and omits the pins.
- Outside the harness I changed only `requires_llm = false` in `task.toml` and one line in `.gitignore`, for the reference measurement cache.

## Running it

```
export FREECAD_CMD=/path/to/freecadcmd
python3 tests/task/harness/harness.py solution/solution.FCStd
python3 tests/task/harness/selftest.py        # scoring logic, no FreeCAD
```

Setting `SLING_KEEP_DIR=<dir>` keeps each measurement. `rescore.py <dir>` then re-scores without FreeCAD.
