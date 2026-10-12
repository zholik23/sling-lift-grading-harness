from __future__ import annotations

import ast
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def _numeric(value):
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def render_dir(task_dir):
    """Where the pictures a judge is shown are kept, for one task.

    Beside `results/`, and not shipped: a record of runs, like the captures
    and the judge cache. Kept rather than thrown away so a verdict can be
    checked afterwards against the thing that was actually looked at --
    the answer is in the capture, and this is the question.

    Twenty-one harnesses each defined this, all to the same path. The task
    directory is passed rather than found because a harness already knows
    its own: it is the one thing the caller has and this module does not.
    """
    return Path(task_dir) / "results" / "renders"


def grading_timeout_s(task_toml, default=900.0):
    """How long one model may be graded for, from `[metadata.harness]`.

    ONE number, in one place. It used to live in three -- task.toml, the
    Harness class and the batch Spec -- and raising it in task.toml alone
    changed nothing, because the batch runner took its own: the reference
    kept being killed at 900 s while the file said 1800.

    Read out of the parsed section, never by scanning lines. Twenty-three
    harnesses scanned for a line starting `timeout_s`, which also matches
    `timeout_sec` -- Harbor's key, in a different section entirely. That
    only ever returned the right number because every task.toml happens to
    write `timeout_s` first, and one task with no `timeout_s` at all was
    already reading Harbor's by accident and agreeing with itself by luck.

    There is no fallback to `timeout_sec`. The two are kept equal in the
    file so that Harbor stops the verifier at the moment the harness
    expects to be stopped; a reader that quietly accepted either would
    hide the day they drift apart, which is the whole failure above.
    """
    try:
        import tomllib
    except ImportError:                                     # py < 3.11
        try:
            import tomli as tomllib                         # type: ignore
        except ImportError:
            return default
    try:
        data = tomllib.loads(Path(task_toml).read_text(encoding="utf-8"))
    except Exception:                                       # noqa: BLE001
        return default
    got = _numeric(((data.get("metadata") or {})
                    .get("harness") or {}).get("timeout_s"))
    return got if got else default


def write_env(var, text):
    """Hand the batch runner what it asked for, if it asked.

    WITHOUT THIS THE LIVE RUN KEEPS NOTHING. `--batch` shells out per
    model and reads back only the envelope, so a run that scored zero
    everywhere left no capture to look at and no report to read: the
    `results/full/` files beside it were the previous OFFLINE run's, and
    they said the drawings had been read perfectly. Two runs, two
    stories, one directory.

    The variable is named in the message rather than the path, because
    the variable is the thing the caller set and can fix.
    """
    dest = os.environ.get(var)
    if not dest:
        return
    try:
        out = Path(dest)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    except Exception as exc:                                # noqa: BLE001
        print(f"  ! could not write {var}: {exc}", file=sys.stderr)


def declared_examples(task_toml):
    """The example folders `task.toml` actually declares.

    THE CORPUS IS WHAT SHIPS, NOT WHAT IS ON DISK. A folder that is
    present but not listed is parked, and a batch that walked the
    directory instead would grade it and report a second model passing,
    which reads from `summary.md` exactly like a corpus with two
    references. Task 67 shipped that mistake once.

    Returns None when the file cannot be read or declares nothing, and
    then every folder is taken -- a task with no `[metadata.examples]`
    block is not a task with no examples. Two of the eight callers
    declare a block; the rest rely on that None.

    Read out of the parsed section. Eight harnesses scanned for lines
    beginning `"examples/` and skipped the ones beginning `#`, which is
    a comment stripper written by hand to do what the parser does for
    free -- and both of the tasks that use this park their folder BY
    commenting the entry out, so the hand-rolled version was the only
    thing standing between a parked model and a second reference.
    """
    try:
        import tomllib
    except ImportError:                                     # py < 3.11
        try:
            import tomli as tomllib                         # type: ignore
        except ImportError:
            return None
    try:
        data = tomllib.loads(Path(task_toml).read_text(encoding="utf-8"))
    except Exception:                                       # noqa: BLE001
        return None
    block = (data.get("metadata") or {}).get("examples") or {}
    want = {str(k).split("/")[1] for k in block
            if str(k).startswith("examples/") and "/" in str(k)}
    return want or None


def true_box(body):
    """A body's real extent, from its tessellation, never from its box.

    `GetBodyBox` returns the box of the UNTRIMMED surfaces. On 67's
    sheath it reads 1644 x 5272 x 17.76 mm for a solid that is
    189 x 59 x 6; the tessellation is the trimmed geometry and reads
    188.834 x 59.052 x 6.000. Each caller's docstring carries its own
    measurement, because the error is a property of the part and not of
    this arithmetic.

    The stored box is used only where no face was tessellated at all --
    a wrong answer being better than no answer nowhere, but a missing
    tessellation means there is nothing else to say.
    """
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    seen = False
    for f in (body.get("faces_all") or []):
        t = f.get("tess")
        if not t or not t.get("extent_mm"):
            continue
        e = t["extent_mm"]
        seen = True
        for i in range(3):
            lo[i] = min(lo[i], e[i])
            hi[i] = max(hi[i], e[i + 3])
    if seen:
        return [round(v, 4) for v in lo + hi]
    return list(body.get("bbox_mm") or [0] * 6)


def judge_cache_dir(task_dir):
    """Where an answer is kept so the same question is not paid for twice.

    One question is one answer: the key is the prompt, so a capture whose
    numbers did not change is not re-asked, and `--rejudge` over fourteen
    captures costs only the ones that actually differ.

    Beside `results/`, and not shipped, for the same reason as the
    renders: a record of runs, not of the task.

    `HARNESS_JUDGE_CACHE` overrides it, so a batch can point every task at
    one cache -- two harnesses already honoured it and the other
    twenty-three ignored it, which is the kind of split that makes a flag
    look broken rather than absent. Twenty-five copies named the directory
    two different ways, `results/judge` seventeen times and
    `results/judge_cache` six; the majority spelling wins and no cache was
    orphaned, because none had been written yet.
    """
    return Path(os.environ.get("HARNESS_JUDGE_CACHE")
                or (Path(task_dir) / "results" / "judge"))


def _rot3(rot, v):
    """A row-major 3x3 as SolidWorks hands it back, applied to a point."""
    return [rot[0] * v[0] + rot[3] * v[1] + rot[6] * v[2],
            rot[1] * v[0] + rot[4] * v[1] + rot[7] * v[2],
            rot[2] * v[0] + rot[5] * v[1] + rot[8] * v[2]]


def aabb(box, rot, tr):
    """Box of a part-space box after the component transform.

    All eight corners go through, so the answer can only be larger than
    the true box -- the safe direction for a filter that must not throw
    away a real overlap.

    Thirteen harnesses carried this. Twelve had dropped the paragraph
    above, which is the only thing that says why the over-estimate is
    deliberate: without it the next reader tightens the bound and the
    filter starts discarding real overlaps, silently and in the safe
    direction's opposite.
    """
    out = [[], [], []]
    for i in (0, 3):
        for j in (1, 4):
            for k in (2, 5):
                q = _rot3(rot, [box[i], box[j], box[k]])
                p = [q[0] + tr[0], q[1] + tr[1], q[2] + tr[2]]
                for n in range(3):
                    out[n].append(p[n])
    return [min(out[0]), min(out[1]), min(out[2]),
            max(out[0]), max(out[1]), max(out[2])]


def decimal_text(value):
    """A number out of text, whichever decimal separator it carries.

    ANNOTATION AND DOCUMENT TEXT IS WRITTEN IN A LOCALE. A drawing saved
    on the Polish SolidWorks the grading box runs writes `0,009` and
    `Ra  0,8`; the same drawing saved elsewhere writes `0.009`. Datasheets
    and reports arrive both ways for the same reason. A candidate is
    graded on the number, never on the separator their install chose, so
    every harness that reads a number out of text goes through here.

    Three harnesses each carried their own `replace(",", ".")` before this
    existed -- 2_shaft_surfaces on finish symbols and geometric tolerance
    frames, 59_metal_grade on DimXpert annotation names, and
    95_thermoplastic_bracket on five separate datasheet patterns -- which
    is three places to remember and three to get wrong.

    Returns None rather than raising, like `_numeric`, so a caller can
    tell "no number here" from a number that happens to be zero.
    """
    if value is None:
        return None
    return _numeric(str(value).strip().replace(",", "."))


def clamp01(value):
    f = _numeric(value)
    return 0.0 if f is None else max(0.0, min(1.0, f))


def score_identity(value):
    return clamp01(value)


def score_error(err, perfect, zero):
    f = _numeric(err)
    if f is None:
        return 0.0
    f = abs(f)
    if f <= perfect:
        return 1.0
    if f >= zero:
        return 0.0
    return 1.0 - (f - perfect) / (zero - perfect)


def score_ratio(value, full, zero=0.0):
    f = _numeric(value)
    if f is None:
        return 0.0
    if f >= full:
        return 1.0
    if f <= zero:
        return 0.0
    return (f - zero) / (full - zero)


score_at_least = score_ratio


def score_band(value, lo, hi, zero_err):
    f = _numeric(value)
    if f is None:
        return 0.0
    if lo <= f <= hi:
        return 1.0
    return score_error((lo - f) if f < lo else (f - hi), 0.0, zero_err)


RUN_FLAG = "--_run"


def positional_argv(argv=None):
    """The arguments that are not switches, argv[0] dropped.

    THE SWITCHES ARE NOT ALWAYS GONE BY THE TIME THESE ARE READ.
    `harness_cli.cli` consumes `--no-judge` and `--no-images` by
    filtering the list it was handed, but the single-model path below
    reads `sys.argv` itself, where they are still sitting. So
    `harness.py --no-judge MODEL.sldasm` put the switch at index 1 and
    the model at index 2, and the model path was parsed as a timeout:

        ValueError: could not convert string to float:
        'solution\\solution.sldasm'

    -- from `float(argv[2])`, on every task in this repository, for a
    switch every task offers. Anything beginning with `--` is dropped
    here; a bare `-` and a negative number are left alone, being
    neither.
    """
    argv = sys.argv if argv is None else argv
    return [a for a in argv[1:] if not str(a).startswith("--")]


def candidate_from_argv(argv=None):
    rest = positional_argv(argv)
    if len(rest) not in (1, 2):
        raise SystemExit("Usage: python3 harness.py candidate.py [timeout_s]")
    return rest[0]


def timeout_from_argv(default, argv=None):
    rest = positional_argv(argv)
    return float(rest[1]) if len(rest) > 1 else default


def is_run_request(argv=None):
    argv = sys.argv if argv is None else argv
    return len(argv) >= 2 and argv[1] == RUN_FLAG


def safe_check(key, descriptions, check_fn, *args):
    try:
        return check_fn(*args)
    except Exception:
        return (key, False, descriptions.get(key, key))


def all_failed(descriptions, **overrides):
    return {
        key: (key, overrides.get(key, False), desc)
        for key, desc in descriptions.items()
    }


RESULT_VAR_NAMES = ("solid", "result", "model", "robot", "part", "final",
                    "assembly", "body", "output", "res")

MIN_SOLID_VOLUME = 1e-9


def _cq():
    import cadquery as cq
    return cq


def is_shape(obj):
    try:
        cq = _cq()
    except ImportError:
        return False
    return isinstance(obj, (cq.Workplane, cq.Shape, cq.Assembly))


def to_shapes(value):
    cq = _cq()
    shapes = []
    if isinstance(value, cq.Workplane):
        for v in value.vals():
            if isinstance(v, cq.Shape) and v.Volume() > MIN_SOLID_VOLUME:
                shapes.append(v)
    elif isinstance(value, cq.Assembly):
        shapes.append(value.toCompound())
    elif isinstance(value, cq.Shape):
        if value.Volume() > MIN_SOLID_VOLUME:
            shapes.append(value)
    return shapes


def pick_result(namespace, shown=(), preferred_names=RESULT_VAR_NAMES):
    chosen = []
    for obj in shown:
        chosen.extend(to_shapes(obj))
    if chosen:
        return chosen, "show_object"

    for name in preferred_names:
        for key, value in namespace.items():
            if key.lower() == name:
                shapes = to_shapes(value)
                if shapes:
                    return shapes, f"variable:{key}"

    best, best_vol, source = [], -1.0, None
    for key, value in namespace.items():
        if key.startswith("_"):
            continue
        try:
            shapes = to_shapes(value)
        except Exception:
            continue
        if not shapes:
            continue
        bb_vol = 0.0
        for s in shapes:
            bb = s.BoundingBox()
            bb_vol += bb.xlen * bb.ylen * bb.zlen
        if bb_vol > best_vol:
            best, best_vol, source = shapes, bb_vol, f"largest:{key}"
    return best, source


def compound(shapes):
    cq = _cq()
    return shapes[0] if len(shapes) == 1 else cq.Compound.makeCompound(shapes)


def safe_volume(shape):
    try:
        return float(shape.Volume())
    except Exception:
        return 0.0


def shape_signature(shape):
    bb = shape.BoundingBox()
    c = shape.Center()
    return {
        "volume": safe_volume(shape),
        "centroid": (c.x, c.y, c.z),
        "bbox": (bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax),
    }


def to_mesh(shape, tolerance=0.05, angular_tolerance=0.2, method="stl"):
    import trimesh

    cq = _cq()
    if method == "tessellate":
        from common import geom

        return geom.to_trimesh(shape, tolerance=tolerance)

    import tempfile

    wp = shape if isinstance(shape, cq.Workplane) else cq.Workplane(obj=shape)
    with tempfile.TemporaryDirectory() as d:
        stl = os.path.join(d, "shape.stl")
        cq.exporters.export(wp, stl, tolerance=tolerance,
                            angularTolerance=angular_tolerance)
        return trimesh.load(stl, force="mesh")


def as_shape(value):
    cq = _cq()
    if isinstance(value, cq.Workplane):
        vals = [x for x in value.vals() if isinstance(x, cq.Shape)]
        if not vals:
            return None
        return vals[0] if len(vals) == 1 else cq.Compound.makeCompound(vals)
    return value if isinstance(value, cq.Shape) else None


ASSEMBLY_NAME_CANDIDATES = ("solid", "result", "assembly", "part", "model")


def pick_named_shape(namespace, name=None, preferred=ASSEMBLY_NAME_CANDIDATES):
    if name:
        shape = as_shape(namespace.get(name))
        if shape is None:
            raise SystemExit(f"'{name}' is not a shape in that script.")
        return name, shape

    best_name, best_shape, best_score = None, None, -1.0
    for key, value in namespace.items():
        if key.startswith("_"):
            continue
        shape = as_shape(value)
        if shape is None:
            continue
        try:
            bb = shape.BoundingBox()
            envelope = ((bb.xmax - bb.xmin) * (bb.ymax - bb.ymin)
                        * (bb.zmax - bb.zmin))
            if float(shape.Volume()) <= 0:
                continue
        except Exception:
            continue
        score = envelope * (1000.0 if key in preferred else 1.0)
        if score > best_score:
            best_name, best_shape, best_score = key, shape, score
    return best_name, best_shape


VIEWER_MODULES = ("jupyter_cadquery", "ocp_vscode", "cq_editor")


def stub_viewer_modules():
    import types as _types

    for name in VIEWER_MODULES:
        if name not in sys.modules:
            m = _types.ModuleType(name)
            m.show = lambda *a, **k: None
            m.show_object = lambda *a, **k: None
            m.set_port = lambda *a, **k: None
            m.reset_show = lambda *a, **k: None
            sys.modules[name] = m


import contextlib


@contextlib.contextmanager
def scratch_cwd():
    """Run candidate code with its working directory pointed at a throwaway
    temp dir, so relative-path side effects (exports, logs) never land in
    the repo. The previous cwd is always restored."""
    import shutil
    import tempfile

    prev = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="candidate_cwd_")
    os.chdir(tmp)
    try:
        yield tmp
    finally:
        os.chdir(prev)
        shutil.rmtree(tmp, ignore_errors=True)


def exec_script(path, overrides=None, module_name="__main__",
                viewer_stubs=False, script_dir_on_path=False):
    if viewer_stubs:
        stub_viewer_modules()

    path = os.path.abspath(str(path))
    ns = {
        "__name__": module_name,
        "__file__": path,
        "show": lambda *a, **k: None,
        "show_object": lambda *a, **k: None,
        "debug": lambda *a, **k: None,
    }

    script_dir = os.path.dirname(path)
    inserted = False
    if script_dir_on_path and script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        inserted = True
    try:
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
        if overrides:
            apply_overrides(tree, overrides)
            ast.fix_missing_locations(tree)
        with scratch_cwd():
            exec(compile(tree, path, "exec"), ns)
    finally:
        if inserted:
            try:
                sys.path.remove(script_dir)
            except ValueError:
                pass
    return ns


def load_script(path, overrides=None):
    path = Path(path)
    ns = {"__name__": "__main__", "__file__": str(path)}
    shown = []
    ns["show_object"] = lambda obj, *a, **k: shown.append(obj)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if overrides:
            apply_overrides(tree, overrides)
            ast.fix_missing_locations(tree)
        with scratch_cwd():
            exec(compile(tree, str(path), "exec"), ns)
    except Exception as exc:
        ns["__shown__"] = shown
        return ns, exc
    ns["__shown__"] = shown
    return ns, None


def apply_overrides(tree, overrides):
    remaining = dict(overrides)
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target]
        for t in targets:
            if t.id in remaining:
                node.value = ast.Constant(value=remaining.pop(t.id))
    if remaining:
        raise RuntimeError(f"Override targets not found: {sorted(remaining)}")


def spawn_run(harness_file, script_path, out_geom, out_meta, timeout,
              overrides=None):
    import time

    cmd = [sys.executable, os.path.abspath(harness_file), RUN_FLAG,
           os.path.abspath(script_path), os.path.abspath(out_geom),
           os.path.abspath(out_meta)]
    if overrides:
        cmd.append(json.dumps(overrides))

    import tempfile

    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="candidate_cwd_") as run_dir:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, cwd=run_dir)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout}s",
                "elapsed": time.time() - t0}
    elapsed = time.time() - t0

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        return {"ok": False, "error": "\n".join(tail), "elapsed": elapsed}
    if not (os.path.exists(out_geom) and os.path.exists(out_meta)):
        return {"ok": False, "error": "runner produced no output geometry",
                "elapsed": elapsed}

    with open(out_meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta.update({"ok": True, "elapsed": elapsed, "geometry": out_geom})
    return meta


def spawn_run_ok(harness_file, script_path, out_geom, out_meta, timeout,
                 overrides=None):
    result = spawn_run(harness_file, script_path, out_geom, out_meta, timeout,
                       overrides=overrides)
    if result.get("ok"):
        return True, None
    return False, result.get("error", "execution failed")


def runner_main(script_path, out_geom, out_meta, overrides_json=None,
                fmt="stl"):
    cq = _cq()

    overrides = json.loads(overrides_json) if overrides_json else None
    ns, err = load_script(script_path, overrides=overrides)
    if err is not None:
        raise err

    chosen, source_of = pick_result(ns, ns.get("__shown__", ()))
    if not chosen:
        raise RuntimeError("Script executed but produced no solid geometry "
                           "(no Workplane/Shape/Assembly with volume found).")

    final = compound(chosen)

    if fmt == "brep":
        final.exportBrep(out_geom)
    else:
        cq.exporters.export(cq.Workplane(obj=final), out_geom,
                            tolerance=0.05, angularTolerance=0.2)

    bb = final.BoundingBox()
    solids = final.Solids()
    meta = {
        "volume": safe_volume(final),
        "bbox": [[bb.xmin, bb.ymin, bb.zmin], [bb.xmax, bb.ymax, bb.zmax]],
        "n_shapes": len(chosen),
        "n_solids": len(solids),
        "volumes": [safe_volume(s) for s in solids],
        "result_source": source_of,
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f)


TOL = 1e-9


def parse(path):
    return ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)


def names(stmt):
    out = []

    def walk_target(t):
        if isinstance(t, ast.Name):
            out.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                walk_target(e)

    if isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            walk_target(t)
    elif isinstance(stmt, ast.AnnAssign):
        walk_target(stmt.target)

    return out


def value_node(stmt):
    if isinstance(stmt, ast.Assign):
        return stmt.value
    if isinstance(stmt, ast.AnnAssign):
        return stmt.value
    return None


def assignments(tree, name):
    return [s for s in tree.body if name in names(s)]


def safe_eval(node, env):
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        return env[node.id]

    if isinstance(node, ast.UnaryOp):
        v = safe_eval(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return v

    if isinstance(node, ast.BinOp):
        a = safe_eval(node.left, env)
        b = safe_eval(node.right, env)

        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return a / b

    raise ValueError("not statically evaluable")


def env_before(tree, line):
    env = {}

    for stmt in tree.body:
        if getattr(stmt, "lineno", 10**9) >= line:
            break

        ns = names(stmt)
        node = value_node(stmt)

        if len(ns) == 1 and node is not None:
            try:
                env[ns[0]] = safe_eval(node, env)
            except Exception:
                pass

    return env


def value(tree, name, last=True):
    xs = assignments(tree, name)
    if not xs:
        raise KeyError(name)

    stmt = xs[-1] if last else xs[0]
    return safe_eval(value_node(stmt), env_before(tree, stmt.lineno))


def close(a, b):
    return abs(float(a) - float(b)) <= TOL


def methods(node, method):
    return [
        n for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == method
    ]


def contains_name(node, name):
    return any(
        isinstance(n, ast.Name) and n.id == name
        for n in ast.walk(node)
    )


def transitively_references(tree, root_name, target_name, _memo=None):
    if _memo is None:
        _memo = set()
    if root_name in _memo:
        return False
    _memo.add(root_name)

    xs = assignments(tree, root_name)
    if not xs:
        return False
    node = value_node(xs[-1])
    if node is None:
        return False
    if contains_name(node, target_name):
        return True

    referenced = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    referenced.discard(root_name)
    return any(transitively_references(tree, r, target_name, _memo) for r in referenced)


def contains_attr(node, attr):
    return any(
        isinstance(n, ast.Attribute) and n.attr == attr
        for n in ast.walk(node)
    )


def normalized_assignment(tree, name):
    xs = assignments(tree, name)
    if not xs:
        return None

    return ast.dump(
        value_node(xs[-1]),
        annotate_fields=True,
        include_attributes=False
    )


def is_numeric_expr(node, known_params=()):
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return is_numeric_expr(node.operand, known_params)
    if isinstance(node, ast.BinOp):
        return (is_numeric_expr(node.left, known_params)
                and is_numeric_expr(node.right, known_params))
    if isinstance(node, ast.Name):
        return node.id in known_params
    return False


def referenced_after(tree, name, after_lineno):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Name) and node.id == name
                and isinstance(node.ctx, ast.Load)
                and getattr(node, "lineno", 0) > after_lineno):
            return True
    return False


def scan_imports_and_attrs(tree):
    imports, attrs = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Attribute):
            attrs.add(node.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            attrs.add(node.func.id)
    return imports, attrs


def load_module(path):
    import runpy

    return runpy.run_path(str(path))


HARNESS_VERSION = "1.0.0"


def _clamp01(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if v != v:
        return 0.0
    return min(1.0, max(0.0, v))


def read_task_id(task_dir):
    """Task id from task.toml's [task].name, minus its "storygold/"
    namespace prefix. task.toml is never bundled into a Harbor verifier
    container, so this stays "" there regardless."""
    task_dir = Path(task_dir)
    try:
        import tomllib
        with open(task_dir / "task.toml", "rb") as f:
            cfg = tomllib.load(f)
        name = cfg["task"]["name"]
        return name.split("/", 1)[1] if "/" in name else name
    except Exception:
        return ""


#: Where a Harbor verifier expects its reward. `/logs/verifier` on Linux,
#: `C:\logs\verifier` on Windows, and `LOGS_DIR` overrides both.
def verifier_log_dir():
    override = os.environ.get("LOGS_DIR")
    if override:
        return Path(override) / "verifier"
    return Path("C:/logs/verifier") if os.name == "nt" else Path("/logs/verifier")


def write_reward(envelope, log_dir=None):
    r"""The one number a Harbor verifier is judged by, where it looks for it.

    Harbor reads `reward.json`, falling back to `reward.txt`, and does not
    read stdout at all. Every harness here printed its envelope to stdout
    and wrote no reward file, so in a verifier run they would all have
    reported nothing -- however well they graded.

    The reward is the score as a FRACTION of the maximum. Raw scores here
    run from 4.0 to 12.0 depending on the task, and a number whose meaning
    changes per task cannot be compared across a dataset.

    Written only when the log directory already exists, which is what
    distinguishes a verifier run from someone running the harness on their
    own machine: creating `C:\logs\verifier` on a workstation because a
    grader happened to run there is litter, not output.
    """
    log_dir = Path(log_dir) if log_dir else verifier_log_dir()
    if not log_dir.parent.is_dir():
        return None
    try:
        score = float(envelope.get("score") or 0.0)
        top = float(envelope.get("max_score") or 0.0)
    except (TypeError, ValueError):
        return None
    reward = round(score / top, 6) if top else 0.0
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "reward.txt").write_text(f"{reward}\n", encoding="utf-8")
    except OSError:
        return None
    return log_dir / "reward.txt"


def finalize(task_dir, checks, version=HARNESS_VERSION, must_pass=(),
             weights=None):
    subscores = {}
    for name, entry in checks.items():
        value = entry[1] if isinstance(entry, (tuple, list)) else entry
        subscores[name] = round(_clamp01(value), 4)
    # Gated (must_pass) checks are score-neutral: they appear in subscores
    # but contribute no points. Failing any gate zeroes the total score.
    gated = set(must_pass)
    ungated = {name: v for name, v in subscores.items() if name not in gated}
    # Optional per-check weights ({check_name: weight}, default 1). Subscores
    # stay clamped to [0, 1]; each ungated check contributes subscore * weight
    # and adds its weight to max_score. Weight 0 contributes nothing.
    weights = weights or {}
    # EVERY EMITTED CHECK MUST BE PRICED. `weights.get(name, 1)` below
    # silently defaults an unlisted criterion to 1, so a criterion added to
    # a harness but not to its ALL_CRITERIA would be scored at full weight
    # while the rubric never mentions it -- and the report would print a
    # table that is not the one grading. One assert here covers every
    # harness in the set, at the single point they all pass through, which
    # is why this does not live in each harness.
    #
    # Only when the harness declares weights at all: the unimplemented
    # stubs carry ALL_CRITERIA = {} and emit no checks, and that pair is
    # consistent.
    if weights and subscores:
        unpriced = sorted(set(subscores) - set(weights))
        if unpriced:
            raise ValueError(
                f"{read_task_id(task_dir)}: these checks are not in "
                f"ALL_CRITERIA and would be scored at a default weight of "
                f"1: {unpriced}")
    score = round(sum(v * weights.get(name, 1)
                      for name, v in ungated.items()), 4)
    failed_must_pass = [name for name in must_pass
                        if subscores.get(name, 1.0) < 1.0]
    if failed_must_pass:
        score = 0.0
    envelope = {
        "task_id": read_task_id(task_dir),
        "score": score,
        # Rounded for the same reason `score` above is: a sum of weights
        # like 2.0 + 1.2 + 1.4 lands on 10.000000000000002 in binary, and
        # that is what a reader of the envelope sees next to a score of
        # 10.0. Six places is far beyond any weight anyone writes and
        # cannot move a comparison.
        "max_score": round(sum(weights.get(name, 1) for name in ungated), 6),
        "passed": bool(subscores) and all(v >= 1.0 for v in subscores.values()),
        "subscores": subscores,
        "harness_version": version,
    }
    if must_pass:
        envelope["must_pass"] = list(must_pass)
    return envelope


# --------------------------------------------------------------------------
# Harness base class
# --------------------------------------------------------------------------

# FreeCAD stage runner (see Harness.run_freecad_stage)

FREECAD_MEASURE_JSON = "measure.json"


def load_dotenv_upwards(start, levels=8):
    """Load the nearest .env at or above `start` (a file or directory), so
    FREECAD_CMD and friends can live in the repo root .env."""
    d = Path(start).resolve()
    if d.is_file():
        d = d.parent
    for _ in range(levels):
        env = d / ".env"
        if env.is_file():
            try:
                import dotenv
            except ImportError:
                return None
            dotenv.load_dotenv(env)
            return env
        d = d.parent
    return None


def freecad_cmd(start=None):
    """Path to freecadcmd (FreeCAD 1.1) from FREECAD_CMD (or FREECADCMD),
    loading the nearest .env first. Exits with a clear message when unset."""
    load_dotenv_upwards(start or Path.cwd())
    cmd = os.environ.get("FREECAD_CMD") or os.environ.get("FREECADCMD")
    if not cmd:
        raise SystemExit("FREECAD_CMD is not set; define it in a .env file "
                         "or the environment (path to freecadcmd 1.1)")
    return cmd


def run_freecad_stage(stage, doc_path, out_dir=None, env=None, timeout=600,
                      freecadcmd=None):
    """Run a FreeCAD measurement stage script under freecadcmd.

    Stage contract (see common/freecad_stage.Stage): the script reads
    FC_STAGE_INPUT (the document to open) and FC_STAGE_OUT (a directory),
    writes `measure.json` there (plus any extra exports it likes, e.g.
    mesh.stl), and sets "ok": false with an "error" string in that JSON to
    report a caught failure. FC_STAGE_COMMON points at the directory holding
    common/ so the script can import the Stage base. Extra env vars for the
    stage go in `env`.

    Returns (measurement_dict, None) on success or (None, error_string).
    When `out_dir` is None a temp dir is used and removed afterwards; pass
    one to keep the stage's extra exports.
    """
    import shutil
    import tempfile

    stage = Path(stage)
    keep = out_dir is not None
    out_dir = Path(out_dir) if keep else Path(tempfile.mkdtemp(prefix="fc_stage_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    full_env = dict(os.environ, FC_STAGE_INPUT=str(Path(doc_path).resolve()),
                    FC_STAGE_OUT=str(out_dir.resolve()),
                    FC_STAGE_COMMON=str(Path(__file__).resolve().parent.parent))
    if env:
        full_env.update({k: str(v) for k, v in env.items()})
    cmd = [freecadcmd or freecad_cmd(stage), str(stage)]
    try:
        try:
            proc = subprocess.run(cmd, env=full_env, capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, f"stage timeout after {timeout}s"
        meas_path = out_dir / FREECAD_MEASURE_JSON
        if not meas_path.is_file():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
            return None, "stage failed: " + " | ".join(tail)
        meas = json.loads(meas_path.read_text(encoding="utf-8"))
        if meas.get("ok") is False:
            return None, "stage failed:\n" + str(meas.get("error", "?"))[-800:]
        return meas, None
    finally:
        if not keep:
            shutil.rmtree(out_dir, ignore_errors=True)


class Harness:
    """Base class for task harnesses: all the scaffolding, none of the checks.

    A task harness subclasses Harness and provides exactly two things:

        class MyTask(Harness):
            SCORING = {...}                 # declarative scored metrics (optional)
            THRESHOLDS = {...}              # named tunables (optional)

            def build_state(self, candidate_path):   # expensive shared setup
                ...
            def checks(self, state):                 # {name: (name, ok, desc)}
                ...

        main = MyTask.as_main()
        if __name__ == "__main__":
            MyTask.cli()

    Everything else is provided here:

    - CLI contract: candidate path from argv[1], optional grading timeout from
      argv[2] (available as self.timeout inside build_state/checks).
    - The --_run self-reinvocation dispatch used by spawn_run.
    - finalize() envelope + the __main__ JSON printing entry point.
    - SCORING-driven scored checks: kind "identity" (score is the metric itself,
      clamped to 1.0 from `full_at`) and kind "error" (1.0 at <= `perfect`,
      linear to 0.0 at `zero_at`), via self.scored_checks(state).

    SCORING entries: {key: {"metric": <state metrics key or None>,
                            "kind": "identity"|"error",
                            "full_at": float          (identity),
                            "perfect": float, "zero_at": float   (error),
                            "desc": str}}
    Override metric_value() for metrics that need computing rather than lookup.

    Set CANDIDATE_OPTIONAL = True for live-document harnesses (e.g. SolidWorks)
    that grade whatever is open when no candidate path is given; build_state
    then receives None.
    """

    THRESHOLDS = {}
    SCORING = {}
    BUILD_TIMEOUT_S = 600
    RUNNER_FMT = "stl"
    CANDIDATE_OPTIONAL = False
    # Check names whose failure zeroes the total score (see finalize).
    MUST_PASS = ()
    # Optional {check_name: weight} for scored checks (default 1 each; see
    # finalize).
    WEIGHTS = {}

    # ------------------------------------------------------------------
    # per-task hooks

    def build_state(self, candidate_path):
        """Expensive shared setup; returns the state dict checks read."""
        raise NotImplementedError

    def checks(self, state):
        """{check_name: (check_name, ok, description)} -- ok is a bool or
        a float in [0, 1]."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # provided scaffolding

    @classmethod
    def harness_file(cls):
        # Prefer the defining module's __file__: inspect.getfile fails on
        # classes from modules loaded without a sys.modules registration.
        mod = sys.modules.get(cls.__module__)
        path = getattr(mod, "__file__", None) or inspect.getfile(cls)
        return str(Path(path).resolve())

    @classmethod
    def task_dir(cls):
        """Directory holding this task's task.toml. The harness may run
        from its original `<task>/harness/harness.py` location or from the
        Harbor-generated `<task>/tests/task/harness/harness.py` copy, which
        sits two directories deeper -- walk up until task.toml turns up
        (never found inside a Harbor verifier container, where task.toml
        isn't bundled; read_task_id() already tolerates that)."""
        d = Path(cls.harness_file()).parent
        for _ in range(5):
            d = d.parent
            if (d / "task.toml").is_file():
                return d
        return Path(cls.harness_file()).parent.parent

    def main(self):
        if is_run_request():
            runner_main(*sys.argv[2:6], fmt=self.RUNNER_FMT)
            raise SystemExit(0)
        if self.CANDIDATE_OPTIONAL and len(sys.argv) == 1:
            candidate = None
        else:
            candidate = candidate_from_argv()
        self.timeout = timeout_from_argv(self.BUILD_TIMEOUT_S)
        state = self.build_state(candidate)
        return finalize(self.task_dir(), self.checks(state),
                        must_pass=self.MUST_PASS, weights=self.WEIGHTS)

    @classmethod
    def as_main(cls):
        """Module-level main() compatible with tools/report.py."""
        def main():
            return cls().main()
        return main

    @classmethod
    def cli(cls):
        print(json.dumps(cls().main(), indent=1))
        raise SystemExit(0)

    # FreeCAD tasks: out-of-process measurement under freecadcmd

    # Per-task stage script (path); defaults to measure_stage.py next to
    # the harness.
    FREECAD_STAGE = None
    FREECAD_STAGE_TIMEOUT_S = 600

    @classmethod
    def freecad_stage(cls):
        if cls.FREECAD_STAGE:
            return Path(cls.FREECAD_STAGE)
        return Path(cls.harness_file()).parent / "measure_stage.py"

    def run_freecad_stage(self, doc_path, out_dir=None, env=None,
                          timeout=None):
        """Measure `doc_path` with this task's stage script under freecadcmd;
        see the module-level run_freecad_stage() for the stage contract.
        Returns (measurement_dict, None) or (None, error_string)."""
        return run_freecad_stage(self.freecad_stage(), doc_path,
                                 out_dir=out_dir, env=env,
                                 timeout=timeout or self.FREECAD_STAGE_TIMEOUT_S)

    # ------------------------------------------------------------------
    # SCORING-driven scored checks

    def metric_value(self, state, key):
        metrics = state.get("metrics")
        if metrics is None:
            return None
        return metrics.get(self.SCORING[key]["metric"])

    def scored_check(self, state, key):
        cfg = self.SCORING[key]
        value = self.metric_value(state, key)
        if cfg["kind"] == "identity":
            score = score_identity(value)
            if score >= cfg["full_at"]:
                score = 1.0
            return (key, score,
                    f"{cfg['desc']} (score = the metric itself, 1.0 from {cfg['full_at']})")
        return (key, score_error(value, cfg["perfect"], cfg["zero_at"]),
                f"{cfg['desc']}: 1.0 at <= {cfg['perfect']}, 0 at {cfg['zero_at']}")

    def scored_checks(self, state):
        return {key: self.scored_check(state, key) for key in self.SCORING}

def require_baseline_keys(payload, required, out_path):
    """Refuse to freeze a baseline that is missing a measurement.

    A re-freeze is the most dangerous routine operation a harness has. It
    looks like maintenance, it succeeds, and if the new payload lost a
    section that some criterion compares against, that criterion quietly
    starts grading on less evidence instead of failing. We paid for this
    once already: `capture_baseline()` stopped writing the modelling census,
    the command reported success, and two thirds of the hygiene criterion
    switched off across a whole run -- visible only as scores that drifted
    rather than broke.

    So the freeze refuses. A loud failure costs one re-run; a silent one
    costs every grade taken until somebody notices.

    Adopted from the parallel refactor's Baseline.REQUIRED_KEYS, which is
    the same idea expressed as a class attribute.
    """
    missing = [k for k in required
               if k not in payload or payload[k] in (None, {}, [])]
    if missing:
        raise RuntimeError(
            f"refusing to freeze {out_path}: payload is missing {missing} -- "
            "criteria compare against these, and a baseline without them "
            "would silently regrade instead of failing loudly")
    return payload

# ---------------------------------------------------------------------------
# scoring weights: one reader, not one per task
# ---------------------------------------------------------------------------

def find_task_toml(start, levels=3):
    """The nearest task.toml at or above `start`.

    Harnesses run from three different depths -- the repo checkout, the
    Harbor-generated tests/task/harness/ copy, and a working directory of
    someone's own choosing -- so the file is searched for rather than
    computed from a fixed number of `..`.
    """
    start = Path(start)
    for d in [start] + list(start.parents)[:levels - 1]:
        if (d / "task.toml").is_file():
            return d / "task.toml"
    return start / "task.toml"


class Baseline:
    """The seed measurement a task grades deltas against.

    Every harness in this repository freezes one and reads it back, and the
    three copies of that code had each solved a different part of the
    problem well and the others not at all -- the same pattern the batch
    runner showed before it moved to harness_cli:

      * task 1 warned when the file's schema was older than the harness
        expects, but let a MISSING file surface as a bare FileNotFoundError;
      * task 15 also warned, and returned None for a missing file, which is
        what freezing needs but not what grading needs;
      * task 17 raised a SystemExit naming the command that creates the file
        -- much the friendliest failure -- but checked no schema at all.

    This takes the better half of each: a missing file fails with the fix
    printed next to it (or returns None when the caller says the baseline is
    optional), and a schema older than the harness warns without refusing,
    because an older baseline is still a valid measurement of the same seed;
    it simply carries fewer probes, and the criteria that used them drop out
    rather than fail.

    `freeze` refuses to write a payload that is missing REQUIRED_KEYS -- see
    require_baseline_keys for why that guard exists and what it cost to
    learn.

    The grader always receives a PLAIN DICT, never an instance: the scoring
    half must stay pure Python with no dependency on this class.
    """

    def __init__(self, path, schema=None, required_keys=()):
        self.path = Path(path)
        self.schema = schema
        self.required_keys = tuple(required_keys)

    def load(self, path=None, optional=False):
        """The frozen baseline as a dict, or None when optional and absent."""
        path = Path(path) if path else self.path
        if not path.is_file():
            if optional:
                return None
            raise SystemExit(
                f"no frozen baseline at {path}. Create it once, with "
                f"SolidWorks running:\n  python harness.py --capture-baseline")
        data = json.loads(path.read_text(encoding="utf-8"))
        got = data.get("schema")
        if self.schema and got and got != self.schema:
            print(f"warning: baseline schema {got!r}, expected "
                  f"{self.schema!r} -- probes missing from it are dropped, "
                  f"not failed", file=sys.stderr)
        return data

    def freeze(self, payload, path=None, dump=None, **extra):
        """Write the payload as the new baseline, or refuse and say why."""
        path = Path(path) if path else self.path
        require_baseline_keys(payload, self.required_keys, path)
        record = dict(payload)
        if self.schema:
            record["schema"] = self.schema
        record.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = dump(record) if dump else json.dumps(record, indent=1,
                                                    default=str)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}" + (f" (schema {self.schema})"
                                 if self.schema else ""), file=sys.stderr)
        return record

# ---------------------------------------------------------------------------
# report assembly: the shape every criterion-scoring task produces
# ---------------------------------------------------------------------------

def part_status(score):
    """PASS / FAIL / PART, the three-letter spelling used in reports.

    Deliberately NOT this module's PASS/FAIL/PARTIAL constants: those spell
    the middle state "PARTIAL", and the reports tasks 15 and 17 have already
    written -- and which we compare byte-for-byte after every refactor --
    say "PART". The spelling is part of the output contract, not a detail to
    tidy on the way past.
    """
    return "PASS" if score >= 0.999 else ("FAIL" if score <= 1e-9 else "PART")


def resolve_weights(weights, defaults, criteria, loader=None):
    """Weights for exactly `criteria`, filled from defaults where task.toml
    is silent. A criterion the file never mentions still gets its designed
    weight rather than 1.0, which would quietly reshape the rubric."""
    if weights is None:
        weights = loader() if loader else {}
    return {k: weights.get(k, defaults[k]) for k in criteria}


def assemble_report(cap, baseline, weights, criteria, analysis, scored,
                    version, status=part_status):
    """The report dict shared by every capture-vs-baseline task.

    Lifted from two BYTE-IDENTICAL copies (tasks 15 and 17); task 1 builds
    a differently shaped report and keeps its own. Extracted at two rather
    than three because byte-identity is the strongest evidence of
    commonality there is, and because task 18 is being written now -- the
    point of the rule of three is to see the shared shape before it is
    copied again, not to wait for the third copy to exist.

    `scored` is {criterion: (score, evidence)}. Every criterion is emitted
    even when the candidate is ungradable, so max_score holds and envelopes
    stay comparable across models.
    """
    report = {
        "harness_version": version,
        "document": cap.get("debug_document"),
        "capture_schema": cap.get("schema"),
        "baseline_schema": (baseline or {}).get("schema"),
        "rebuild": cap.get("rebuild"),
        "weights": weights,
        "criteria": {k: {"score": round(scored[k][0], 4),
                         "weight": weights[k],
                         "status": status(scored[k][0]),
                         "evidence": scored[k][1]} for k in criteria},
        "measurements": analysis,
    }
    report["overall_score"] = round(
        sum(report["criteria"][k]["score"] * weights[k] for k in criteria), 4)
    if analysis.get("ungradable"):
        report["ungradable"] = analysis["ungradable"]
    if analysis.get("notes"):
        report["notes"] = analysis["notes"]
    return report

class Grader:
    """One shape for every capture-vs-baseline grader in this repository.

    Named for what it produces rather than what it knows: task 1 already has
    a `Grader` full of gamepad geometry, and two classes called Grader in one
    codebase is how a reader ends up in the wrong file.

    A subclass supplies exactly two things:

        analyse()  -> the measurements dict every criterion reads. Computed
                      ONCE in __init__, so criteria share it instead of each
                      recomputing the same body match.
        score()    -> {criterion: (score01, evidence)} from self.m

    and declares CRITERIA, DEFAULT_WEIGHTS and VERSION. Everything after
    that -- resolving weights against task.toml, emitting every criterion
    even for an ungradable candidate, the weighted total, the report keys --
    is the same for all of them and lives here.

    Why a class rather than the pair of free functions tasks 15 and 17 used:
    not because those duplicated anything (they did not -- `analyse` already
    computed shared state once), but because four harnesses are read
    together by whoever reviews them, and a reader who has understood one
    should not have to re-derive the shape of the next. Uniformity is a
    property of the deliverable, not of the codebase.

    The report is a PLAIN DICT. Nothing downstream depends on this class.

    ONE TASK DELIBERATELY DOES NOT USE THIS, and the reason is worth having
    written down rather than rediscovered. Task 1 (the PlayStation
    controller) grades a PART, and its grader carries a rebuild GATE: when
    the feature tree does not rebuild clean it zeroes the geometry criteria,
    keeps health and hygiene, and returns an overall that is the MEAN of the
    criteria rather than their weighted sum. assemble_report() cannot
    produce that shape, and bending it until it could would change numbers
    that are calibrated against a corpus.

    So the family is: tasks 15, 17 and everything assembly-shaped after them
    subclass this; task 1 keeps its own grader and says so. Uniformity that
    has to lie about a real difference is worth less than the difference
    being visible.
    """

    CRITERIA = ()
    DEFAULT_WEIGHTS = {}
    VERSION = None
    STATUS = staticmethod(part_status)

    def __init__(self, baseline, capture):
        self.baseline = baseline or {}
        self.capture = capture or {}
        #: shared measurements, computed once for every criterion
        self.m = self.analyse()

    def analyse(self):
        raise NotImplementedError

    def score(self):
        raise NotImplementedError

    def ungradable(self):
        """The reason this candidate could not be MEASURED, or None.

        Distinct from measured-and-wrong: an ungradable candidate still
        emits every criterion, so max_score holds and envelopes stay
        comparable across a batch."""
        return self.m.get("ungradable")

    def report(self, weights=None, loader=None):
        weights = resolve_weights(weights, self.DEFAULT_WEIGHTS,
                                  self.CRITERIA, loader)
        reason = self.ungradable()
        scored = ({k: (0.0, f"UNGRADABLE: {reason}") for k in self.CRITERIA}
                  if reason else self.score())
        return assemble_report(self.capture, self.baseline, weights,
                               self.CRITERIA, self.m, scored, self.VERSION,
                               status=self.STATUS)
