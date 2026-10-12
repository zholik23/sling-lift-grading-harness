#!/usr/bin/env python3
"""Self-test for the scoring half, no FreeCAD needed.

    python3 tests/task/harness/selftest.py

Builds a toy pantograph lift as a measurement (measure.json + assembly.stl,
exactly what measure_stage.py writes), derives each adversarial from it the
way the README describes, plus honest variants that model the same lift
differently, and runs the real harness over all of them with the toy as the
reference. It checks the scoring logic, not measure_stage.py: that half
needs freecadcmd and the real corpus.
"""
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent


def seg_box(p, q, t=20.0, y=0.0):
    p, q = np.array(p, float), np.array(q, float)
    d = q - p
    L = np.linalg.norm(d) + 20.0
    m = trimesh.creation.box((L, t, t))
    ang = math.atan2(d[2], d[0])
    m.apply_transform(trimesh.transformations.rotation_matrix(-ang, [0, 1, 0]))
    m.apply_translation((p + q) / 2 + [0, y, 0])
    return m


def box(c, e):
    m = trimesh.creation.box(e)
    m.apply_translation(c)
    return m


def solid(m):
    return {"V": float(m.volume), "A": float(m.area),
            "com": [float(x) for x in m.center_mass],
            "I": [float(x) for x in np.asarray(m.moment_inertia).ravel()]}


def frame(p, theta=0.0):
    """Joint frame at p with Z along world Y, rotated theta about it."""
    R = trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])  # z->y
    R = R @ trimesh.transformations.rotation_matrix(math.radians(theta), [0, 0, 1])
    R[:3, 3] = p
    return R


def build_reference():
    P0, PL, PR, PB = (0, 0, 560), (-200, 0, 220), (200, 0, 220), (0, 0, 0)
    meshes = {
        "ring": box((0, 0, 637.5), (65, 16, 75)),
        "hub": box((0, 0, 570), (60, 60, 60)),
        "armL": seg_box(P0, PL, y=0),
        "armR": seg_box(P0, PR, y=20),
        "linkL": seg_box(PL, PB, y=0),
        "linkR": seg_box(PR, PB, y=20),
        "clutchL": box((-200, 0, 150), (40, 30, 120)),
        "clutchR": box((200, 20, 150), (40, 30, 120)),
    }
    names = list(meshes)
    bodies = []
    for n in names:
        m = meshes[n]
        bodies.append({"name": n, "label": n, "type": "PartDesign::Body",
                       "solids": [solid(m)], "volume": float(m.volume),
                       "diag": float(np.linalg.norm(m.extents)),
                       "extent": [float(x) for x in m.extents],
                       "cyl_d": [16.0, 20.0] if n in ("hub", "armL") else [],
                       "axis_d": [120.0] if n == "armR" else [],
                       "dead_volume": 0.0})
    ix = {n: i for i, n in enumerate(names)}

    def J(name, t, a, b, p, th1=0.0, th2=0.0, limits=None):
        g1, g2 = frame(p, th1), frame(p, th2)
        return {"name": name, "type": t, "active": True, "suppressed": False,
                "b1": ix[a], "b2": ix[b],
                "g1": g1.ravel().tolist(), "g2": g2.ravel().tolist(),
                "g1o": g1.ravel().tolist(), "g2o": g2.ravel().tolist(),
                "anchor1": 0.0, "anchor2": 0.0, "offset1_mag": 0.0,
                "offset2_mag": 0.0, "limits": limits or {}, "distance": 0.0,
                "n_refs": 2}
    joints = [
        J("w", "Fixed", "ring", "hub", (0, 0, 600)),
        J("j1", "Revolute", "hub", "armL", P0, 0, 20, {"AngleMin": 0.0, "AngleMax": 60.0}),
        J("j2", "Revolute", "hub", "armR", (0, 10, 560), 0, -20, {"AngleMin": -60.0, "AngleMax": 0.0}),
        J("j3", "Revolute", "armL", "linkL", PL),
        J("j4", "Revolute", "armR", "linkR", (200, 10, 220)),
        J("j5", "Revolute", "linkL", "linkR", (0, 10, 0)),
        J("j6", "Revolute", "armL", "clutchL", (-200, 0, 210)),
        J("j7", "Revolute", "armR", "clutchR", (200, 20, 210)),
    ]
    sig = {"H": 687, "Hs": 511, "W": 453, "Wc": 216, "F": 228.5,
           "Lk": 120, "Rl": 100, "Rw": 65}
    carriers = [{"kind": "sheet", "owner": "Params", "key": f"B{i}",
                 "value": float(v), "alt": [], "named": True, "unit": "mm",
                 "alias": k} for i, (k, v) in enumerate(sig.items())]
    carriers += [{"kind": "sketch", "owner": "Sk", "key": i, "value": 30.0 + i,
                  "alt": [], "named": False, "unit": "mm"} for i in range(8)]
    meas = {
        "ok": True, "assembly": True, "recompute_errors": [],
        "bodies": bodies, "joints": joints, "grounded": [ix["hub"]],
        "sketches": [{"name": f"S{i}", "used": True, "geo_dof": 16,
                      "removed_dof": 16, "fully": True, "n_dims": 3,
                      "n_named": 1} for i in range(8)],
        "carriers": carriers,
        "probes": [{"carrier": i, "changed": 1.0} for i in range(len(carriers))],
        "mesh": {"file": "assembly.stl"},
    }
    return meas, meshes


# ------------------------------------------------------------------ variants

def transform_body(meas, meshes, name, T):
    """Move one body rigidly: its mesh, its solids, and the joint frames on it."""
    i = [b["name"] for b in meas["bodies"]].index(name)
    meshes[name] = meshes[name].copy()
    meshes[name].apply_transform(T)
    meas["bodies"][i]["solids"] = [solid(meshes[name])]
    for j in meas["joints"]:
        for k in ("1", "2"):
            if j["b" + k] == i:
                for key in ("g" + k, "g%so" % k):
                    j[key] = (T @ np.array(j[key]).reshape(4, 4)).ravel().tolist()


def drop_body(meas, meshes, name):
    i = [b["name"] for b in meas["bodies"]].index(name)
    del meas["bodies"][i]
    del meshes[name]
    keep = []
    for j in meas["joints"]:
        if i in (j["b1"], j["b2"]):
            continue
        for k in ("b1", "b2"):
            if j[k] > i:
                j[k] -= 1
        keep.append(j)
    meas["joints"] = keep
    meas["grounded"] = [g - (g > i) for g in meas["grounded"] if g != i]
    return i


def half_finished(meas, meshes):
    drop_body(meas, meshes, "ring")
    for b in meas["bodies"]:
        b["cyl_d"] = []                                   # holes never cut
    meas["carriers"] = [c for c in meas["carriers"] if c.get("alias") not in ("Rl", "Rw")]
    meas["probes"] = [{"carrier": i, "changed": 1.0} for i in range(len(meas["carriers"]))]


def one_link_missing(meas, meshes):
    drop_body(meas, meshes, "linkR")
    names = [b["name"] for b in meas["bodies"]]
    j = copy.deepcopy(next(x for x in meas["joints"] if x["name"] == "j3"))
    j.update(name="bridge", b1=names.index("armR"), b2=names.index("linkL"),
             offset1_mag=283.0, anchor1=0.0, anchor2=0.0)
    # the joint "mates" only because of a 283 mm offset across the gap
    g = np.array(j["g1"]).reshape(4, 4)
    g[:3, 3] = (0, 10, 0)
    j["g2"] = j["g2o"] = j["g1o"] = g.ravel().tolist()
    meas["joints"].append(j)


def dummy_joints(meas, meshes):
    real = meas["joints"]
    out = []
    for k, j in enumerate(real * 3):
        d = copy.deepcopy(j)
        d["name"] = f"dummy{k}"
        if k % 3 == 0:
            d["b2"] = d["b1"]                             # self-referencing
        elif k % 3 == 1:
            d["active"] = False                           # switched off
        else:
            d["b1"] = d["b2"] = None                      # references nothing
        out.append(d)
    meas["joints"] = out


def joint_limits_swapped(meas, meshes):
    for j in meas["joints"]:
        lim = j["limits"]
        if "AngleMin" in lim:
            lim["AngleMin"], lim["AngleMax"] = lim["AngleMax"], lim["AngleMin"]
    # with no valid range the solver leaves the arms folded up to one side
    T = trimesh.transformations.rotation_matrix(math.radians(75), [0, 1, 0], (0, 0, 560))
    for n in ("armL", "armR", "linkL", "linkR", "clutchL", "clutchR"):
        transform_body(meas, meshes, n, T)


def mirror(meas, meshes):
    M = np.diag([-1.0, 1, 1, 1])
    F = np.diag([-1.0, 1, 1, 1])                          # re-handedness the frame
    for n in list(meshes):
        m = meshes[n].copy()
        m.apply_transform(M)          # trimesh re-winds reflected faces itself
        meshes[n] = m
        b = next(b for b in meas["bodies"] if b["name"] == n)
        b["solids"] = [solid(m)]
    for j in meas["joints"]:
        for key in ("g1", "g2", "g1o", "g2o"):
            j[key] = (M @ np.array(j[key]).reshape(4, 4) @ F).ravel().tolist()
        lim = j["limits"]
        if "AngleMin" in lim:
            lim["AngleMin"], lim["AngleMax"] = -lim["AngleMax"], -lim["AngleMin"]


def alt_modeller(meas, meshes):
    """Same lift, built differently: rotated and moved in space, bodies in a
    different order with different names, joint Z axes the other way (so
    limits change sign), the hub as two solids, parameters in sketches
    under other names."""
    T = trimesh.transformations.rotation_matrix(math.radians(90), [0, 0, 1])
    T[:3, 3] = (500, -300, 40)
    for n in list(meshes):
        transform_body(meas, meshes, n, T)
    flip = trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0])
    for j in meas["joints"]:
        for key in ("g1", "g2", "g1o", "g2o"):
            j[key] = (np.array(j[key]).reshape(4, 4) @ flip).ravel().tolist()
        lim = j["limits"]
        if "AngleMin" in lim:
            lim["AngleMin"], lim["AngleMax"] = -lim["AngleMax"], -lim["AngleMin"]
    hub = next(b for b in meas["bodies"] if b["name"] == "hub")
    m = meshes["hub"]
    lo, hi = m.bounds
    mid = (lo + hi) / 2
    halves = [trimesh.creation.box(((hi - lo)[0], (hi - lo)[1], (hi - lo)[2] / 2))
              for _ in range(2)]
    halves[0].apply_translation(mid - [0, 0, (hi - lo)[2] / 4])
    halves[1].apply_translation(mid + [0, 0, (hi - lo)[2] / 4])
    hub["solids"] = [solid(h) for h in halves]
    order = list(range(len(meas["bodies"])))[::-1]
    inv = {old: new for new, old in enumerate(order)}
    meas["bodies"] = [meas["bodies"][o] for o in order]
    for k, b in enumerate(meas["bodies"]):
        b["label"] = b["name"] = f"Part{k:03d}"
    for j in meas["joints"]:
        j["b1"], j["b2"] = inv[j["b1"]], inv[j["b2"]]
        j["name"] = "Joint" + j["name"]
    meas["grounded"] = [inv[g] for g in meas["grounded"]]
    for c in meas["carriers"]:
        c.update(kind="sketch", owner="SketchX", alias=None)
    meas["joints"].reverse()
    # meshes keyed by old name; keep the mapping for writing
    return {meshes_key: meshes_key for meshes_key in meshes}


def alt_no_limits(meas, meshes):
    """Honest build that sets no joint limits at all."""
    for j in meas["joints"]:
        j["limits"] = {}


def decorative_dims(meas, meshes):
    meas["carriers"] += [{"kind": "sheet", "owner": "Deco", "key": f"Z{i}",
                          "value": 10.0 + i, "alt": [], "named": True,
                          "unit": "mm"} for i in range(25)]
    n0 = len(meas["carriers"]) - 25
    meas["probes"] = ([{"carrier": n0 + i, "changed": 0.0} for i in range(10)]
                      + [{"carrier": i, "changed": 0.0} for i in range(8, 14)])


def double_scale(meas, meshes):
    S = np.diag([2.0, 2, 2, 1])
    for n in list(meshes):
        m = meshes[n].copy()
        m.apply_transform(S)
        meshes[n] = m
        b = next(b for b in meas["bodies"] if b["name"] == n)
        b["solids"] = [solid(m)]
        b["volume"] = float(m.volume)
        b["diag"] *= 2
        b["extent"] = [2 * x for x in b["extent"]]
        b["cyl_d"] = [2 * x for x in b["cyl_d"]]
        b["axis_d"] = [2 * x for x in b["axis_d"]]
    for j in meas["joints"]:
        for key in ("g1", "g2", "g1o", "g2o"):
            g = np.array(j[key]).reshape(4, 4)
            g[:3, 3] *= 2
            j[key] = g.ravel().tolist()
    for c in meas["carriers"]:
        c["value"] *= 2


VARIANTS = {
    "reference (itself)": None,
    "alt modeller (rotated, renamed, reordered, flipped axes, split hub)": alt_modeller,
    "alt modeller with no joint limits": alt_no_limits,
    "half_finished_missing_features": half_finished,
    "one_link_missing_joints_bridge_gap": one_link_missing,
    "dummy_joints_inflate_the_count": dummy_joints,
    "joint_limits_swapped": joint_limits_swapped,
    "assembly_mirrored_relative_to_ground_truth": mirror,
    "decorative_dims (from task.toml notes)": decorative_dims,
    "double_scale (from task.toml notes)": double_scale,
}


def write(d, meas, meshes):
    d.mkdir(parents=True, exist_ok=True)
    (d / "measure.json").write_text(json.dumps(meas))
    trimesh.util.concatenate(list(meshes.values())).export(str(d / "assembly.stl"))


def main():
    tmp = Path(tempfile.mkdtemp(prefix="sling_selftest_"))
    ref_meas, ref_meshes = build_reference()
    write(tmp / "reference", ref_meas, ref_meshes)
    env = dict(os.environ, SLING_REFERENCE=str(tmp / "reference"))
    rows = []
    for label, fn in VARIANTS.items():
        meas, meshes = copy.deepcopy(ref_meas), {k: v.copy() for k, v in ref_meshes.items()}
        if fn:
            fn(meas, meshes)
        d = tmp / label.split(" ")[0]
        write(d, meas, meshes)
        res = subprocess.run([sys.executable, str(HERE / "harness.py"), str(d)],
                             env=env, capture_output=True, text=True)
        try:
            env_out = json.loads(res.stdout)
        except json.JSONDecodeError:
            print(res.stdout, res.stderr)
            raise
        rows.append((label, env_out))
    names = list(rows[0][1]["subscores"])
    short = ["gate", "geom", "mated", "artic", "cons", "drive", "sig"]
    print(f"{'model':<70} {'score':>6}  " + " ".join(f"{s:>5}" for s in short))
    for label, e in rows:
        print(f"{label[:70]:<70} {e['score']:>6.2f}  " +
              " ".join(f"{e['subscores'][n]:>5.2f}" for n in names))
    if "-v" in sys.argv:
        for label, e in rows:
            print("\n==", label)
            print(json.dumps(e, indent=1))


if __name__ == "__main__":
    main()
