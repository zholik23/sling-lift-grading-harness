# Sling Lift — FreeCAD grading harness

Submission for the StoryGold **CAD Evaluation Engineer** take-home: a grader for `FreeCAD/4_sling_lift`. The original assignment brief is in [ASSIGNMENT.md](ASSIGNMENT.md).

**The write-up** (what is measured, what is let go, the mirror decision, scores): [`FreeCAD/4_sling_lift/tests/task/harness/NOTES.md`](FreeCAD/4_sling_lift/tests/task/harness/NOTES.md)

## Results on the real corpus (FreeCAD 1.1.4)

| Model | Score | Loses on |
|---|---|---|
| reference | **6.00 / 6** | — |
| assembly_mirrored | 6.00 | — (a mirror is treated as a valid answer; detected and reported) |
| half_finished_missing_features | 5.86 | geometry |
| joint_limits_swapped | 5.50 | mechanism articulates |
| one_link_missing_joints_bridge_gap | 5.17 | joints mated, mechanism articulates |
| dummy_joints_inflate_the_count | 4.33 | joints mated, mechanism articulates |

Each example keeps 1.0 on every component it doesn't break. The full score envelopes, with the sentence behind each number, are in [`results/`](results/).

## What's here

| Path | Contents |
|---|---|
| `FreeCAD/4_sling_lift/tests/task/harness/harness.py` | Entry point. Never imports FreeCAD; runs the stage under `freecadcmd` and scores the JSON. |
| `.../harness/measure_stage.py` | Runs inside `freecadcmd`: parts, joints, sketches, dimension carriers, perturbation probes. |
| `.../harness/scoring.py` | Pure numpy/scipy scoring; testable without FreeCAD. |
| `.../harness/selftest.py` | Synthetic corpus, including honest re-modellings that must score 6/6. |
| `.../harness/rescore.py` | Re-scores saved measurements without FreeCAD. |
| `results/` | Score envelopes from the final run. |

## Running

```
pip install -r env_requirements.txt
python3 tools/fetch.py                       # the .FCStd models live in Azure Blob Storage
export FREECAD_CMD=/path/to/freecadcmd       # FreeCAD 1.1
cd FreeCAD/4_sling_lift
python3 tests/task/harness/harness.py solution/solution.FCStd
python3 tests/task/harness/selftest.py       # no FreeCAD needed
```
