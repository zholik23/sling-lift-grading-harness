# Sling Lift — a FreeCAD grading harness

Welcome, and thanks for taking this on.

This repo holds one CAD task. Someone is asked to rebuild a sling lift in
FreeCAD from a drawing. **Your job is to write the thing that grades their
answer.**

`harness.py` and `measure_stage.py` are your starting point. They run,
and they print a proper score sheet, so you can see the shape of what
you're filling in. The thinking is all yours.

There's no answer key in the repo, and a couple of the judgement calls
are genuinely arguable. We're more interested in how you reason about
them than in whether you land where we did.

## Deliverable

Two things, and they pull against each other:

1. The reference answer, `solution/solution.FCStd`, scores **6 out of 6**.
2. Each of the five broken models in `examples/` loses points **only** on
   what it actually gets wrong.

The second one is where it gets interesting. It's easy to write a grader
that fails everything, and easy to write one that passes everything. The
hard part is a grader that's right for the right reasons.

So grade the *geometry* and the *parametric structure* the brief asks
for. Don't grade object names, or the shape of the document tree, or
anything that just happens to be true of our reference file. Someone who
models the lift differently and gets it right should still score 6 of 6.

When you're done, send us the harness plus a short note: what you chose
to measure, what you decided to let go, and the scores you got over the
reference and all five examples.

## Setup

The FreeCAD documents and their `.stl` meshes live in Azure Blob Storage
rather than git, so a fresh clone doesn't have them yet. The `.png`
renders are committed, so you can see the whole corpus first.

```bash
pip install -r env_requirements.txt
python3 tools/fetch.py
```

That pulls down anything missing or out of date. Then point `FREECAD_CMD`
at your `freecadcmd` binary:

```bash
export FREECAD_CMD=/usr/bin/freecadcmd
```

On Windows that path looks more like `.../FreeCAD 1.1/bin/freecadcmd.exe`.
You can also put the line in a `.env` at the repo root and the harness
will find it.

Good news on setup: this one is **headless**. No Windows, no GUI, no
licence. If FreeCAD 1.1 runs on your machine, you're set.

```bash
cd FreeCAD/4_sling_lift
python3 tests/task/harness/harness.py solution/solution.FCStd
```

Right now that prints zeros. That's expected. It's the stub.

## The task

`FreeCAD/4_sling_lift/instruction.md` has the exact wording. The gist:

> Convert a Sling Lift 30 kg assembly to parametric CAD, driven by key
> dimensions rather than dead geometry, with the pantograph joints kept
> mated so it articulates.

One thing worth noticing: the input is **a PDF drawing**, not a model.
Whoever does this task builds the document from nothing. So there's no
starting file for your harness to diff against. You're comparing a
candidate to a reference, not to an input.

## Test corpus

Five broken models live in `examples/`, each built to catch a
particular shortcut. Think of them as the test suite for your grader.

| Model | What's wrong with it |
|---|---|
| `half_finished_missing_features` | Stops partway. Features the drawing asks for just aren't there. |
| `one_link_missing_joints_bridge_gap` | A link is missing, and the joints stretch to cover the gap. |
| `dummy_joints_inflate_the_count` | Plenty of joints. None of them constrain anything. |
| `joint_limits_swapped` | Joints are there and mated, but their limits are swapped. |
| `assembly_mirrored_relative_to_ground_truth` | Geometrically fine, but mirrored against our reference. |

The first two catch a grader that never really compares against the
drawing, one that checks the file is well-formed and calls it a day.

`dummy_joints` catches a grader that counts joints instead of asking what
they do. `joint_limits_swapped` catches one that checks joints exist and
are mated but never asks what they permit.

Then there's `assembly_mirrored`, which is the one we'd most like to hear
your thinking on. Its geometry is arguably *correct*, just reflected. A
grader that compares against our reference pose without allowing for a
reflection will fail it. But that same grader would also fail an honest
candidate who happened to build the lift the other way round.

So: is the mirror a defect, or is our reference just one of two valid
answers? We have an opinion. We're not going to tell you what it is. Make
a call and defend it.

## Harness architecture

Measurement happens **in a separate process**, and it's worth keeping it
that way.

`harness.py` never imports FreeCAD. Instead it runs `measure_stage.py`
under `freecadcmd`, which opens the document, measures it, and writes
`measure.json`. Your harness reads that JSON and turns it into scores.

Two reasons this split is nice to keep. Your scoring logic stays testable
on a machine with no FreeCAD installed. And a candidate's document never
gets to run its own macros inside your interpreter, which matters more
than it sounds like it should.

`measure_stage.py` is empty on purpose. What you pull out of the document
decides what you're able to grade, so that choice is most of the job
rather than a detail to sort out afterwards.

The harness prints a JSON envelope with `score`, `max_score`, `passed`
and `subscores`, built by `finalize()` in `common/harness_base.py`. These
are the components it ships with:

| Component | Weight |
|---|---:|
| executes and builds geometry | gate (0) |
| geometry matches ground truth | 1 |
| pantograph joints mated | 1 |
| mechanism articulates | 1 |
| carries dimensional constraints | 1 |
| dimensions drive the geometry | 1 |
| signature dimensions present | 1 |

Treat that list as a starting suggestion, not a spec. Split them, merge
them, reweight them, rename them, throw some out, whatever your reading
of the brief supports. Just keep `max_score` in `task.toml` equal to the
sum, and tell us what you changed.

The gate is already written, by the way. "Executes and builds geometry"
is the one thing a stub can answer honestly, and leaving it in means a
genuinely broken document still looks different from an unwritten
harness.

### Scoring

Each component returns a number between 0 and 1. A bool is accepted, and
it's almost always the wrong answer.

Most of these are questions of degree, and collapsing them to pass/fail
throws away exactly the information that makes a grader useful:

| Situation | Good | Not so good |
|---|---:|---:|
| five joints wanted, three mated | 0.6 | 0 |
| a link 4% short of the drawing | 0.9 | 0 |
| two of four signature dimensions present | 0.5 | 0 |

Someone who gets most of it right should get most of the marks, and two
candidates who are wrong by different amounts shouldn't land on the same
number. Save a hard 0 or 1 for the components that genuinely have no
middle — "does it rebuild without errors" is one, and that's the gate.

`common/harness_base.py` has helpers for the usual shapes: `clamp01`,
`score_error(value, perfect, zero)` for a reading that degrades as it
drifts from a target, `score_ratio` for "how many of these are right",
and `score_band` for a value that has to sit inside a range.

## LLM judge (optional)

`common/llm_judge.py` is here and working, and nothing in the stub calls
it.

Some questions are awkward to measure and easy to ask. "Does this
actually read as a pantograph?" is that kind of question. If you decide a
component is better judged than measured, subclass `Judge` and give it
`AZURE_API_KEY` and `AZURE_CLAUDE_RESOURCE` in your `.env`.

One thing it does deliberately: it raises rather than scoring when it
can't reach a model. An unavailable judge shouldn't quietly become a
pass.

Entirely your call. A fully deterministic harness is a fine answer, and
so is a hybrid. We just want to hear why.

## Repository layout

| Path | What it is |
|---|---|
| `instruction.md` | The prompt the candidate sees. |
| `task.toml` | Task metadata, asset URLs and checksums, environment config. |
| `environment/` | `input.pdf`, the drawing they work from, plus a Dockerfile. |
| `solution/` | The reference answer, and `solve.sh`, which just copies it into place. |
| `examples/` | The five broken models, one folder each. |
| `tests/task/harness/` | `harness.py` and `measure_stage.py`. Both stubs. Both yours. |
| `tests/task/reference/` | A frozen copy of the reference, plus an `.stl` and `.png` per model. |
| `tests/test.sh` | The verifier entrypoint. Runs your harness, writes `reward.json`. |

Shared code lives in `common/` at the repo root: `harness_base.py` (the
`Harness` base class, the stage runner, `finalize`), `freecad_stage.py`
(the `Stage` base your measurement half subclasses), and `geom.py`.

## Validation

Run it over the reference and all five examples:

```bash
cd FreeCAD/4_sling_lift
python3 tests/task/harness/harness.py solution/solution.FCStd
python3 tests/task/harness/harness.py examples/adversarial_joint_limits_swapped/adversarial_joint_limits_swapped.FCStd
```

The reference should come back 6 of 6. Each example should lose exactly
the components it breaks, and keep the rest.

### Docker

If you'd rather work in Docker:

```bash
docker build -f common/docker/freecad-base.Dockerfile -t freecad-base:latest .
docker build -f FreeCAD/4_sling_lift/environment/Dockerfile -t sling-lift FreeCAD/4_sling_lift
```

`tests/test.sh` is the entrypoint there. It expects the candidate at
`/app/solution.FCStd` and writes `/logs/verifier/reward.json`.

---

Questions are welcome. If something in here is ambiguous, that's useful
for us to know, so do ask.
