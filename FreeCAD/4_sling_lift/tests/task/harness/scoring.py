"""Scoring for 4_sling_lift: pure numpy/scipy, no FreeCAD.

Everything here takes two measurements (candidate, reference) in the shape
measure_stage.py writes and returns component scores in [0, 1] with the
sentence a human reads next to each. Nothing reads an object name, a tree
shape or a label: bodies are matched by mass properties, joints by the
bodies they connect, poses by best rigid alignment.

Two principles run through all of it.

1. REFERENCE-RELATIVE, NEVER REFERENCE-SPECIFIC. Where a quantity has a
   natural "how much" (how constrained the sketches are, what fraction of
   probed dimensions move the solids, how many pantograph pivots exist,
   how many of the drawing's signature dimensions are realised) the
   candidate is measured the same way the reference is and normalised by
   it, capped at 1. So the reference scores 1 by construction, and a
   candidate is never asked to match an incidental property of the
   reference file -- only to be at least as parametric, as mated and as
   faithful as it is.

2. INVARIANT TO WHAT THE BRIEF DOES NOT FIX. Body signatures use volume,
   area and principal radii of gyration: invariant to rigid motion, to
   reflection and to articulation. Joint limits are compared by width and
   by where the delivered pose sits in the range, both independent of
   which way a joint frame's Z axis was chosen. Pose comparison takes the
   best alignment over all proper AND improper orthogonal frames, unless
   MIRROR_IS_DEFECT is set.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

#: The one judgement call that is a switch. See NOTES.md for the argument.
#: False: a reflected lift is one of two valid answers and is not penalised.
MIRROR_IS_DEFECT = False

#: Key dimensions read off the drawing (mm). Only those the reference
#: itself exhibits are expected of a candidate -- see signature_dims().
SIGNATURE_DIMS = (687.0, 511.0, 453.0, 216.0, 228.5, 120.0, 100.0, 65.0,
                  16.0, 20.0)

ARTICULATING = ("Revolute", "Cylindrical", "Slider", "Ball")


# --------------------------------------------------------------------------
# small helpers (mirroring common.harness_base so this file stands alone)

def clamp01(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if not math.isfinite(x) else max(0.0, min(1.0, x))


def score_error(err, perfect, zero):
    err = abs(float(err))
    if err <= perfect:
        return 1.0
    if err >= zero:
        return 0.0
    return 1.0 - (err - perfect) / (zero - perfect)


def score_ratio(value, full, zero=0.0):
    value = float(value)
    if value >= full:
        return 1.0
    if value <= zero:
        return 0.0
    return (value - zero) / (full - zero)


def rel_to_ref(cand, ref):
    """Candidate quantity normalised by the reference's, capped at 1."""
    if ref is None or ref <= 1e-9:
        return 1.0 if (cand or 0) > 0 or ref is not None else 0.0
    return clamp01((cand or 0.0) / ref)


def m4(a):
    return np.asarray(a, dtype=float).reshape(4, 4)


# --------------------------------------------------------------------------
# model: bodies, rigid groups, joints -- derived once per measurement

class Model:
    def __init__(self, meas, mesh_dir=None):
        self.meas = meas or {}
        self.mesh_dir = Path(mesh_dir) if mesh_dir else None
        self.bodies = [b for b in self.meas.get("bodies", []) if b]
        self.joints = copy.deepcopy(list(self.meas.get("joints", [])))
        self.grounded = set(self.meas.get("grounded", []))
        n = len(self.bodies)
        self.size = max([b.get("diag", 0.0) for b in self.bodies] + [1.0])
        self.total_volume = sum(b.get("volume", 0.0) for b in self.bodies)
        for j in self.joints:
            self._classify(j)
        # rigid groups: bodies welded by effective Fixed joints
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for j in self.joints:
            if j["_usable"] and j["type"] == "Fixed":
                parent[find(j["b1"])] = find(j["b2"])
        self.group_of = [find(i) for i in range(n)]
        for j in self.joints:
            j["_effective"] = j["_usable"] and (
                j["type"] == "Fixed" or
                self.group_of[j["b1"]] != self.group_of[j["b2"]])
        self._dedupe()
        self.groups = {}
        for i, g in enumerate(self.group_of):
            self.groups.setdefault(g, []).append(i)

    # -- joints ---------------------------------------------------------
    def _classify(self, j):
        n = len(self.bodies)
        b1, b2 = j.get("b1"), j.get("b2")
        ok_refs = (isinstance(b1, int) and isinstance(b2, int)
                   and 0 <= b1 < n and 0 <= b2 < n and b1 != b2)
        j["_usable"] = bool(j.get("active", True) and not j.get("suppressed")
                            and ok_refs and constraint_rows(j["type"]) > 0)
        j["_anchored"] = False
        j["_mated"] = False
        if not ok_refs:
            return
        d1 = self.bodies[b1].get("diag", self.size)
        d2 = self.bodies[b2].get("diag", self.size)
        tol_a = lambda d: max(1.0, 0.01 * d)              # noqa: E731
        a1, a2 = j.get("anchor1"), j.get("anchor2")
        off_tol = max(5.0, 0.05 * min(d1, d2))
        j["_anchored"] = (a1 is not None and a2 is not None
                          and a1 <= tol_a(d1) and a2 <= tol_a(d2)
                          and j.get("offset1_mag", 0.0) <= off_tol
                          and j.get("offset2_mag", 0.0) <= off_tol)
        j["_mated"] = (frames_mated(j["type"], j.get("g1o"), j.get("g2o"), self.size)
                       or frames_mated(j["type"], j.get("g1"), j.get("g2"), self.size))

    def _dedupe(self):
        seen = []
        for j in self.joints:
            j["_duplicate"] = False
            if not j["_effective"]:
                continue
            key = (frozenset((self.group_of[j["b1"]], self.group_of[j["b2"]])),
                   j["type"])
            p, z = frame_axis(j.get("g1o") or j.get("g1"))
            for k, p2, z2 in seen:
                if k == key and np.linalg.norm(p - p2) < 0.002 * self.size \
                        and abs(abs(z @ z2) - 1) < 1e-3:
                    j["_duplicate"] = True
                    break
            if not j["_duplicate"]:
                seen.append((key, p, z))

    def honest(self, j):
        return (j["_effective"] and not j["_duplicate"]
                and j["_anchored"] and j["_mated"])

    def articulating(self):
        return [j for j in self.joints if j["type"] in ARTICULATING
                and j["_effective"] and not j["_duplicate"]]

    # -- signatures -----------------------------------------------------
    def solid_sigs(self):
        out = []
        for b in self.bodies:
            for s in b.get("solids", []):
                sig = mass_signature([s])
                if sig is not None:
                    out.append(sig)
        return out

    def group_sigs(self, level="groups"):
        """Signatures per welded rigid group, or per body ('bodies')."""
        if level == "bodies":
            groups = {i: [i] for i in range(len(self.bodies))}
        else:
            groups = self.groups
        out = {}
        for g, members in groups.items():
            solids = [s for i in members for s in self.bodies[i].get("solids", [])]
            sig = mass_signature(solids)
            if sig is not None:
                out[g] = sig
        return out

    # -- mechanism ------------------------------------------------------
    def mechanism(self):
        """Bodies kinematically tied to ground (or the biggest cluster)."""
        n = len(self.bodies)
        adj = {i: set() for i in range(n)}
        for j in self.joints:
            if j["_effective"]:
                adj[j["b1"]].add(j["b2"])
                adj[j["b2"]].add(j["b1"])
        comps, seen = [], set()
        for i in range(n):
            if i in seen:
                continue
            stack, comp = [i], set()
            while stack:
                k = stack.pop()
                if k in comp:
                    continue
                comp.add(k)
                stack.extend(adj[k] - comp)
            seen |= comp
            comps.append(comp)
        jointed = [c for c in comps if len(c) > 1]
        if not jointed:
            return set()
        grounded = [c for c in jointed if c & self.grounded]
        pool = grounded or jointed
        return max(pool, key=lambda c: sum(self.bodies[i].get("volume", 0) for i in c))

    def mobility(self, comp):
        if not comp:
            return 0
        idx = {b: k for k, b in enumerate(sorted(comp))}
        L = self.size
        rows = []
        for j in self.joints:
            if not j["_effective"] or j["b1"] not in idx:
                continue
            for r in joint_rows(j, L):
                row = np.zeros(6 * len(idx))
                i1, i2 = idx[j["b1"]], idx[j["b2"]]
                row[6 * i1:6 * i1 + 6] += r
                row[6 * i2:6 * i2 + 6] -= r
                rows.append(row)
        gauge = 6
        for b in comp & self.grounded:
            gauge = 0
            for k in range(6):
                row = np.zeros(6 * len(idx))
                row[6 * idx[b] + k] = 1.0
                rows.append(row)
        if not rows:
            return 6 * len(idx) - gauge
        J = np.array(rows)
        sv = np.linalg.svd(J, compute_uv=False)
        rank = int((sv > 1e-7 * max(sv.max(), 1e-12)).sum())
        return int(6 * len(idx) - rank - gauge)

    def mesh_points(self, n=20000, seed=0):
        info = self.meas.get("mesh") or {}
        if not info or self.mesh_dir is None:
            return None
        path = self.mesh_dir / info.get("file", "assembly.stl")
        if not path.is_file():
            return None
        import trimesh
        mesh = trimesh.load(str(path), force="mesh")
        if mesh.is_empty or mesh.area <= 0:
            return None
        pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
        return np.asarray(pts, dtype=float)


# --------------------------------------------------------------------------
# geometry primitives

def mass_signature(solids):
    """(V, A, r1, r2, r3): volume, area, sorted principal radii of gyration
    about the combined centre of mass. Invariant to rigid motion and to
    reflection; computed per rigid group so articulation cannot move it."""
    solids = [s for s in solids if s and s.get("V", 0) > 1e-9]
    if not solids:
        return None
    V = sum(s["V"] for s in solids)
    A = sum(s["A"] for s in solids)
    com = sum(s["V"] * np.asarray(s["com"]) for s in solids) / V
    I = np.zeros((3, 3))
    for s in solids:
        d = np.asarray(s["com"]) - com
        I += np.asarray(s["I"], dtype=float).reshape(3, 3) + \
            s["V"] * (d @ d * np.eye(3) - np.outer(d, d))
    ev = np.sort(np.clip(np.linalg.eigvalsh(0.5 * (I + I.T)), 0, None))
    return np.array([V, A, *np.sqrt(ev / V)])


def sig_cost(a, b):
    """Log-distance on volume and principal radii. Area is carried in the
    signature but not compared: splitting one solid into two welded ones
    adds internal faces, so area moves under honest re-modelling while
    volume and inertia do not."""
    la, lb = np.log(np.maximum(a, 1e-9)), np.log(np.maximum(b, 1e-9))
    d = np.abs(la - lb)
    return float(max(d[0], d[2:].mean()))


def match_sigs(cand, ref):
    """Hungarian match of signature lists. Returns per-ref and per-cand
    similarity in [0, 1] (0 for unmatched)."""
    from scipy.optimize import linear_sum_assignment
    sr, sc = np.zeros(len(ref)), np.zeros(len(cand))
    pairs = {}
    if not cand or not ref:
        return sr, sc, pairs
    C = np.array([[sig_cost(c, r) for r in ref] for c in cand])
    ci, ri = linear_sum_assignment(C)
    for c, r in zip(ci, ri):
        s = score_error(C[c, r], 0.01, 0.15)
        sr[r], sc[c] = s, s
        if s > 0:
            pairs[c] = (r, s)
    return sr, sc, pairs


def weighted_fraction(scores, sigs):
    if len(sigs) == 0:
        return 0.0
    w = np.array([s[0] ** (1 / 3) for s in sigs])
    return float((w * scores).sum() / w.sum())


def frame_axis(g):
    if g is None:
        return np.zeros(3), np.array([0, 0, 1.0])
    M = m4(g)
    z = M[:3, 2]
    return M[:3, 3], z / max(np.linalg.norm(z), 1e-12)


def frames_mated(jtype, g1, g2, size):
    if g1 is None or g2 is None:
        return False
    p1, z1 = frame_axis(g1)
    p2, z2 = frame_axis(g2)
    tol = max(0.5, 0.001 * size)
    parallel = abs(abs(z1 @ z2) - 1.0) < 1.6e-4        # ~1 degree
    d = p2 - p1
    off_axis = np.linalg.norm(d - (d @ z1) * z1)
    if jtype in ("Revolute", "Fixed"):
        return parallel and np.linalg.norm(d) <= tol
    if jtype in ("Cylindrical", "Slider"):
        return parallel and off_axis <= tol
    if jtype == "Ball":
        return np.linalg.norm(d) <= tol
    return True                     # Distance/Angle/...: not a mate claim


def constraint_rows(jtype):
    return {"Fixed": 6, "Revolute": 5, "Cylindrical": 4, "Slider": 5,
            "Ball": 3, "Distance": 1, "Parallel": 2, "Perpendicular": 1,
            "Angle": 1}.get(jtype, 0)


def joint_rows(j, L):
    """Linearised constraint rows on one body's twist [w, v] (positions
    scaled by L so the matrix is well conditioned)."""
    g = j.get("g1o") or j.get("g1")
    if g is None:
        return []
    M = m4(g)
    p = M[:3, 3] / L
    x, y, z = (M[:3, k] / max(np.linalg.norm(M[:3, k]), 1e-12) for k in range(3))
    E = (np.eye(3)[0], np.eye(3)[1], np.eye(3)[2])
    pt = lambda e: np.concatenate([np.cross(p, e), e])   # noqa: E731
    rot = lambda e: np.concatenate([e, np.zeros(3)])     # noqa: E731
    t = j["type"]
    if t == "Fixed":
        return [rot(e) for e in E] + [pt(e) for e in E]
    if t == "Revolute":
        return [pt(e) for e in E] + [rot(x), rot(y)]
    if t == "Cylindrical":
        return [rot(x), rot(y), pt(x), pt(y)]
    if t == "Slider":
        return [rot(e) for e in E] + [pt(x), pt(y)]
    if t == "Ball":
        return [pt(e) for e in E]
    if t == "Distance":
        return [pt(z)]
    if t == "Parallel":
        return [rot(x), rot(y)]
    if t in ("Perpendicular", "Angle"):
        return [rot(x)]
    return []


def joint_value(j):
    """Current joint coordinate: angle about Z (deg) and slide along Z."""
    g1, g2 = j.get("g1o") or j.get("g1"), j.get("g2o") or j.get("g2")
    if g1 is None or g2 is None:
        return None, None
    R = np.linalg.inv(m4(g1)) @ m4(g2)
    return math.degrees(math.atan2(R[1, 0], R[0, 0])), float(R[2, 3])


# --------------------------------------------------------------------------
# pose comparison

def _orthos():
    import itertools
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for r, (c, s) in enumerate(zip(perm, signs)):
                M[r, c] = s
            mats.append(M)
    return mats


def _icp(src, tree, dst, R, t, iters=12, trim=0.9, allow_reflection=False):
    for _ in range(iters):
        moved = src @ R.T + t
        d, idx = tree.query(moved)
        keep = d <= np.quantile(d, trim)
        a, b = src[keep], dst[idx[keep]]
        ca, cb = a.mean(0), b.mean(0)
        H = (a - ca).T @ (b - cb)
        U, _, Vt = np.linalg.svd(H)
        D = np.eye(3)
        target_det = np.sign(np.linalg.det(R)) if not allow_reflection else None
        if target_det is not None and np.sign(np.linalg.det(Vt.T @ U.T)) != target_det:
            D[2, 2] = -1
        R = Vt.T @ D @ U.T
        t = cb - ca @ R.T
    return R, t


def align(cand, ref, seed=0):
    """Best rigid alignment of candidate points onto reference points,
    separately for proper rotations and for reflections. Returns
    {"proper": (fscore, R, t), "improper": (...)}"""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    sub = cand[rng.choice(len(cand), min(4000, len(cand)), replace=False)]
    tree = cKDTree(ref)
    cc, rc = cand.mean(0), ref.mean(0)
    _, _, Vc = np.linalg.svd(cand - cc, full_matrices=False)
    _, _, Vr = np.linalg.svd(ref - rc, full_matrices=False)
    tau = 0.01 * float(np.linalg.norm(ref.max(0) - ref.min(0)))
    best = {}
    for S in _orthos():
        R0 = Vr.T @ S @ Vc
        t0 = rc - cc @ R0.T
        R, t = _icp(sub, tree, ref, R0, t0)
        kind = "proper" if np.linalg.det(R) > 0 else "improper"
        d, _ = tree.query(sub @ R.T + t)
        err = float(np.mean(np.minimum(d, 5 * tau)))
        if kind not in best or err < best[kind][0]:
            best[kind] = (err, R, t)
    out = {}
    ctree = None
    for kind, (_, R, t) in best.items():
        moved = cand @ R.T + t
        ctree = cKDTree(moved)
        prec = float((tree.query(moved)[0] < tau).mean())
        rec = float((ctree.query(ref)[0] < tau).mean())
        f = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        out[kind] = (f, R, t)
    return out, tau


def pca_extents(pts):
    c = pts - pts.mean(0)
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    proj = c @ Vt.T
    return list(proj.max(0) - proj.min(0)), list(pts.max(0) - pts.min(0))


# --------------------------------------------------------------------------
# components

def volume_fraction(scores, sigs):
    """Match quality weighted by part volume: the share of the material
    that is accounted for."""
    if len(sigs) == 0:
        return 0.0
    w = np.array([s[0] for s in sigs])
    return float((w * scores).sum() / w.sum())


#: How much of the part set (by volume) may differ from the reference
#: before geometry loses anything, and where it reaches zero. The drawing
#: dimensions the assembly, not every part in it -- e.g. it leaves the
#: hub tube's length open -- so a candidate is not held to parts the
#: drawing does not define. Missing whole parts or features costs.
GEOM_FREE, GEOM_ZERO = 0.05, 0.20


def geometry(cand: Model, ref: Model, cpts, rpts, aligned=None):
    """Are the right parts there, the right shape and size?

    Judged part by part, POSE-FREE: each rigid part's volume and principal
    radii of gyration, which do not change when the mechanism moves, when
    the lift is mirrored, or when parts are named or ordered differently.
    Where the mechanism sits is not a geometric property of an articulating
    assembly; whether it can reach the drawn configuration is judged under
    'mechanism articulates'. Read at three levels -- single solids, bodies,
    welded groups -- because candidates split and merge parts differently;
    the best reading describes the model."""
    levels = {}
    for name, cs, rs in (("solids", cand.solid_sigs(), ref.solid_sigs()),
                         ("bodies", list(cand.group_sigs("bodies").values()),
                          list(ref.group_sigs("bodies").values())),
                         ("welded-group", list(cand.group_sigs().values()),
                          list(ref.group_sigs().values()))):
        sr, sc, _ = match_sigs(cs, rs)
        recall = volume_fraction(sr, rs)
        precision = volume_fraction(sc, cs) if cs else 0.0
        levels[name] = (0.8 * recall + 0.2 * precision, recall, precision)
    lvl = max(levels, key=lambda k: levels[k][0])
    part, recall, precision = levels[lvl]
    score = score_error(1.0 - part, GEOM_FREE, GEOM_ZERO)
    txt = (f"part set matches the reference by mass signature ({lvl}-level): "
           f"{recall:.1%} of the reference's material accounted for, "
           f"{precision:.1%} of the candidate's; combined {part:.3f}. "
           f"Free up to {GEOM_FREE:.0%} difference, zero at {GEOM_ZERO:.0%}")
    # The pose is not graded here, but a reflection is reported -- and
    # penalised only under MIRROR_IS_DEFECT.
    if aligned:
        res, _ = aligned
        f_p = res.get("proper", (0.0,))[0]
        f_m = res.get("improper", (0.0,))[0]
        if f_m > f_p + 0.05:
            txt += (f". Assembled shape fits the reference best as a "
                    f"REFLECTION (F {f_m:.2f} vs {f_p:.2f})")
            if MIRROR_IS_DEFECT:
                score *= 0.5
                txt += ": penalised (MIRROR_IS_DEFECT)"
            else:
                txt += ": not penalised"
    return score, txt


def _joint_pairs(model, j, gmap):
    """The pair of body CLASSES a joint connects (unordered, as a multiset)."""
    if gmap.get("_level") == "bodies":
        a, b = j["b1"], j["b2"]
    else:
        a, b = model.group_of[j["b1"]], model.group_of[j["b2"]]
    a, b = gmap.get(a), gmap.get(b)
    return None if a is None or b is None else tuple(sorted((a, b)))


def ref_classes(ref: Model, level="groups"):
    """Reference rigid groups clustered into classes of identical shape.

    A lift has twins -- left and right clutch, paired links, every bolt --
    and which twin a candidate's body is matched to is arbitrary. Joints
    are therefore compared by the CLASSES of the bodies they connect, so a
    candidate that numbers its left and right parts the other way round
    still matches pivot for pivot."""
    rg = ref.group_sigs(level)
    keys = list(rg)
    cls = {k: k for k in keys}

    def find(k):
        while cls[k] != k:
            k = cls[k]
        return k
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if sig_cost(rg[a], rg[b]) <= 0.02:
                cls[find(a)] = find(b)
    return {k: find(k) for k in keys}


def group_map(cand: Model, ref: Model, level="groups"):
    """(candidate group -> reference class, reference group -> class).

    level "groups" matches welded rigid groups (robust to a part modelled
    as several welded bodies); "bodies" matches single bodies (robust to a
    welded neighbour being missing or modelled separately). Scorers that
    use the map try both and keep the better reading."""
    rcls = ref_classes(ref, level)
    cg, rg = cand.group_sigs(level), ref.group_sigs(level)
    ck, rk = list(cg), list(rg)
    _, _, pairs = match_sigs([cg[k] for k in ck], [rg[k] for k in rk])
    cmap = {ck[c]: rcls[rk[r]] for c, (r, s) in pairs.items() if s >= 0.5}
    cmap["_level"] = rcls["_level"] = level
    return cmap, rcls


def joints_mated(cand: Model, ref: Model, gmap):
    ref_js = [j for j in ref.articulating() if ref.honest(j)]
    n_ref = len(ref_js)
    if n_ref == 0:
        return 0.0, "the reference has no honestly mated pivots to compare against"
    want = {}
    for j in ref_js:
        key = (_joint_pairs(ref, j, gmap[1]), j["type"] in ("Revolute", "Cylindrical"))
        want[key] = want.get(key, 0) + 1
    honest = [j for j in cand.articulating() if cand.honest(j)]
    matched, mappable = 0, 0
    for j in honest:
        pair = _joint_pairs(cand, j, gmap[0])
        mappable += pair is not None
        key = (pair, j["type"] in ("Revolute", "Cylindrical"))
        if want.get(key, 0) > 0:
            want[key] -= 1
            matched += 1
    by_match = matched / n_ref
    # Counting is only a fallback for when the candidate's parts could not
    # be matched to the reference's at all (a very different breakdown into
    # bodies); when they could, the match is the evidence and a count
    # would only reward joints that connect the wrong things.
    by_count = 0.9 * min(1.0, len(honest) / n_ref) \
        if honest and mappable < 0.5 * len(honest) else 0.0
    cj = cand.joints
    stats = dict(
        total=len(cj),
        inactive=sum(1 for j in cj if not j["_usable"]),
        welded=sum(1 for j in cj if j["_usable"] and not j["_effective"]),
        dup=sum(1 for j in cj if j.get("_duplicate")),
        unanchored=sum(1 for j in cand.articulating() if not j["_anchored"]),
        unmated=sum(1 for j in cand.articulating() if not j["_mated"]))
    return max(by_match, by_count), (
        f"{matched} of the reference's {n_ref} pivots are realised by a joint "
        f"between the corresponding bodies that is active, constrains two "
        f"different rigid bodies, sits on both bodies' material with no "
        f"gap-bridging offset, and is coincident at the delivered pose "
        f"({len(honest)} such joints in all). Candidate joints: {stats['total']}, "
        f"of which {stats['inactive']} inactive or unresolved, {stats['welded']} "
        f"between already-welded bodies, {stats['dup']} duplicates, "
        f"{stats['unanchored']} off the material or bridged by an offset, "
        f"{stats['unmated']} not coincident")


def _limit_ranges(j):
    lim = j.get("limits") or {}
    out = []
    for lo_k, hi_k, unit in (("AngleMin", "AngleMax", "deg"),
                             ("LengthMin", "LengthMax", "mm")):
        lo, hi = lim.get(lo_k), lim.get(hi_k)
        if lo is not None or hi is not None:
            out.append((lo, hi, unit))
    return out


def _group_com(model, i):
    sol = model.bodies[i].get("solids", [])
    V = sum(x["V"] for x in sol)
    if V <= 0:
        return None
    return sum(x["V"] * np.asarray(x["com"]) for x in sol) / V


def reaches_reference(cand: Model, ref: Model, j, gmap, pose):
    """Can this slide joint's permitted stroke reach the configuration the
    reference (the drawing) is in?

    Asked physically, so it does not depend on how the candidate chose its
    joint frames or numbered its bodies: the permitted DISPLACEMENT of body
    2 relative to body 1 along the joint axis is [min - cur, max - cur]; the
    displacement NEEDED is how far body 2 must slide, relative to body 1,
    to sit where the matching reference bodies sit -- both compared in the
    reference's space through the best rigid alignment (a reflection
    included, when that is the better fit). Returns (ok, needed, lo, hi)
    or None when it cannot be asked."""
    if pose is None or j["type"] not in ("Slider", "Cylindrical"):
        return None
    lim = j.get("limits") or {}
    lo, hi = lim.get("LengthMin"), lim.get("LengthMax")
    _, cur = joint_value(j)
    if cur is None or (lo is None and hi is None):
        return None
    R, t = pose
    cmap, rcls = gmap
    level = cmap.get("_level")
    out = []
    for b in (j["b1"], j["b2"]):
        key = b if level == "bodies" else cand.group_of[b]
        cls = cmap.get(key)
        com = _group_com(cand, b)
        if cls is None or com is None:
            return None
        mapped = R @ com + t
        members = [i for i in range(len(ref.bodies))
                   if rcls.get(i if level == "bodies" else ref.group_of[i]) == cls]
        coms = [(i, _group_com(ref, i)) for i in members]
        coms = [(i, c) for i, c in coms if c is not None]
        if not coms:
            return None
        i_best, c_best = min(coms, key=lambda ic: np.linalg.norm(ic[1] - mapped))
        out.append((mapped, c_best))
    (c1, r1), (c2, r2) = out
    z = R @ m4(j.get("g1o") or j.get("g1"))[:3, 2]
    needed = float(((r2 - r1) - (c2 - c1)) @ z)
    dlo = -np.inf if lo is None else lo - cur
    dhi = np.inf if hi is None else hi - cur
    tol = max(2.0, 0.01 * ref.size)
    return dlo - tol <= needed <= dhi + tol, needed, dlo, dhi


def joint_limits(cand: Model, ref: Model, gmap, pose=None):
    """What the joints PERMIT, compared in frame-orientation-free terms."""
    ref_by_pair = {}
    for j in ref.articulating():
        if _limit_ranges(j):
            ref_by_pair.setdefault(_joint_pairs(ref, j, gmap[1]), []).append(j)
    scores, notes = [], []
    for j in cand.articulating():
        ranges = _limit_ranges(j)
        if not ranges:
            continue
        ang, slide = joint_value(j)
        s = 1.0
        for lo, hi, unit in ranges:
            cur = ang if unit == "deg" else slide
            tol = 1.0 if unit == "deg" else 0.5
            if lo is not None and hi is not None and lo > hi + 1e-9:
                s = 0.0
                notes.append(f"{j['name']}: min {lo:g} > max {hi:g}")
                continue
            if cur is not None:
                # the joint coordinate and its limits are measured in the
                # same frame, so they are compared with their signs
                inside = (lo is None or cur >= lo - tol) and (hi is None or cur <= hi + tol)
                if not inside:
                    s *= 0.25
                    notes.append(f"{j['name']}: delivered pose {cur:.1f} {unit} "
                                 f"outside [{lo}, {hi}]")
            if unit == "mm":
                reach = reaches_reference(cand, ref, j, gmap, pose)
                if reach is not None and not reach[0]:
                    s = 0.0
                    notes.append(f"{j['name']}: its stroke permits a slide of "
                                 f"{reach[2]:+.0f}..{reach[3]:+.0f} mm from the delivered "
                                 f"pose, but reaching the drawn configuration needs "
                                 f"{reach[1]:+.0f} mm")
            refs = ref_by_pair.get(_joint_pairs(cand, j, gmap[0])) or []
            if not refs or lo is None or hi is None:
                continue
            best, best_note = None, None
            for rj in refs:          # twins share a class: take the best fit
                for rlo, rhi, ru in _limit_ranges(rj):
                    if ru != unit or rlo is None or rhi is None or rhi <= rlo:
                        continue
                    w, rw = hi - lo, rhi - rlo
                    sr = score_error((w - rw) / rw, 0.05, 0.5)
                    note = None if sr >= 1 else f"{j['name']}: range {w:g} vs reference {rw:g}"
                    rcur = joint_value(rj)[0 if unit == "deg" else 1]
                    if cur is not None and rcur is not None and w > 0:
                        # where the pose sits in the range, either sense
                        f, fr = (cur - lo) / w, (rcur - rlo) / rw
                        fm = ((-cur) - lo) / w
                        err = min(abs(f - fr), abs(f - (1 - fr)),
                                  abs(fm - fr), abs(fm - (1 - fr)))
                        sr *= score_error(err, 0.05, 0.4)
                    if best is None or sr > best:
                        best, best_note = sr, note
            if best is not None:
                s *= best
                if best_note:
                    notes.append(best_note)
        scores.append(s)
    if not scores:
        return 1.0, "no joint limits set (the brief does not require them)"
    return float(np.mean(scores)), (
        f"{len(scores)} limited joints, mean {np.mean(scores):.2f}"
        + (": " + "; ".join(notes[:4]) if notes else ": ranges well-formed and "
           "consistent with the reference"))


def articulates(cand: Model, ref: Model, gmap, pose=None):
    rmech, cmech = ref.mechanism(), cand.mechanism()
    m_ref, m_c = ref.mobility(rmech), cand.mobility(cmech)
    # share of the model's own solid volume that is in the jointed
    # mechanism: unit- and scale-free, so a wrongly scaled lift is
    # convicted by geometry, not twice
    rv = sum(ref.bodies[i].get("volume", 0) for i in rmech) / max(ref.total_volume, 1e-9)
    cv = sum(cand.bodies[i].get("volume", 0) for i in cmech) / max(cand.total_volume, 1e-9)
    vol = score_error(max(0.0, 1 - cv / rv), 0.10, 0.70) if rv > 0 else (1.0 if cv > 0 else 0.0)
    if m_c <= 0:
        mob = 0.0
    elif m_ref <= 0:
        mob = score_error(m_c, 0, 6)
    elif m_c < m_ref:
        mob = m_c / m_ref
    else:
        # one spare DoF is allowed for a pin modelled as free to spin;
        # beyond that, every extra DoF is a mechanism that is not the one
        # drawn (a missing link frees the scissor)
        mob = score_error(max(0, m_c - m_ref - 1), 0, 4)
    lim, lim_txt = joint_limits(cand, ref, gmap, pose)
    score = vol * mob * (0.5 + 0.5 * lim)
    return score, (
        f"jointed mechanism holds {cv / max(rv, 1e-9):.0%} of the reference "
        f"mechanism's volume -> {vol:.2f}; instantaneous mobility from the joint "
        f"Jacobian {m_c} vs reference {m_ref} (1 spare spin allowed) -> {mob:.2f}; "
        f"limits: {lim_txt} -> {lim:.2f}. score = volume x mobility x (0.5 + 0.5 x limits)")


def _constrainedness(model):
    sk = [s for s in model.meas.get("sketches", []) if s.get("used")]
    tot = sum(max(s.get("geo_dof", 0), 0) for s in sk)
    if not sk or tot == 0:
        return None, 0
    acc = 0.0
    for s in sk:
        g = max(s.get("geo_dof", 0), 0)
        if s.get("fully") is True:
            c = 1.0
        else:
            c = min(0.95 if s.get("fully") is False else 1.0,
                    s.get("removed_dof", 0) / g if g else 1.0)
        acc += g * c
    return acc / tot, len(sk)


def _live(model):
    tot = model.total_volume
    dead = sum(b.get("dead_volume", 0.0) for b in model.bodies)
    return 0.0 if tot <= 0 else clamp01(1 - dead / tot)


def carries_constraints(cand: Model, ref: Model):
    lc, lr = _live(cand), _live(ref)
    cc, nc = _constrainedness(cand)
    cr, _ = _constrainedness(ref)
    n_car = len(cand.meas.get("carriers", []))
    if cc is None:
        cc = 1.0 if n_car > 0 and lc > 0 else 0.0
    s_live = rel_to_ref(lc, lr)
    s_con = rel_to_ref(cc, cr) if cr is not None else cc
    return s_live * s_con, (
        f"{lc:.0%} of solid volume has modelling history (reference {lr:.0%}) "
        f"-> {s_live:.2f}; {nc} sketches feeding solids are {cc:.0%} constrained "
        f"(reference {('%.0f%%' % (100 * cr)) if cr is not None else 'n/a'}) "
        f"-> {s_con:.2f}; {n_car} dimension carriers")


def _drive_fraction(model):
    cars = model.meas.get("carriers", [])
    pr = model.meas.get("probes", [])
    named = [p["changed"] for p in pr if cars[p["carrier"]].get("named")]
    plain = [p["changed"] for p in pr if not cars[p["carrier"]].get("named")]
    if not pr:
        return None, 0, (0, 0)
    fn = float(np.mean(named)) if named else None
    fp = float(np.mean(plain)) if plain else None
    if fn is not None and fp is not None:
        f = 0.6 * fn + 0.4 * fp
    else:
        f = fn if fn is not None else fp
    return f, len(pr), (fn, fp)


def dims_drive(cand: Model, ref: Model):
    fc, n, (fn, fp) = _drive_fraction(cand)
    fr, _, _ = _drive_fraction(ref)
    if fc is None:
        return 0.0, "no dimension carriers to probe: nothing could drive the geometry"
    s = rel_to_ref(fc, fr) if fr is not None else fc
    fmt = lambda x: "n/a" if x is None else f"{x:.0%}"     # noqa: E731
    return s, (f"{n} carriers nudged 5% and recomputed: named ones moved the "
               f"solids {fmt(fn)} of the time, unnamed {fmt(fp)} -> {fc:.2f} "
               f"(reference {fmt(fr)})"
               + ("; probe budget ran out" if cand.meas.get("probe_budget_hit") else ""))


def _present(v, model, mesh_ext):
    tol_c = max(0.05, 0.001 * v)
    tol_g = max(0.3, 0.005 * v)
    for c in model.meas.get("carriers", []):
        if c.get("unit") == "mm" and any(abs(x - v) <= tol_c
                                         for x in [c["value"], *c.get("alt", [])]):
            return "parameter"
    for b in model.bodies:
        if any(abs(d - v) <= tol_g for d in b.get("cyl_d", [])):
            return "diameter"
        if any(abs(d - v) <= tol_g for d in b.get("axis_d", [])):
            return "pin spacing"
        if any(abs(d - v) <= tol_g for d in b.get("extent", [])):
            return "part extent"
    if any(abs(d - v) <= tol_g for d in mesh_ext):
        return "overall extent"
    return None


def signature_dims(cand: Model, ref: Model, cpts, rpts):
    def ext(p):
        if p is None:
            return []
        a, b = pca_extents(p)
        return a + b
    expected = [v for v in SIGNATURE_DIMS if _present(v, ref, ext(rpts))] \
        or list(SIGNATURE_DIMS)
    cext = ext(cpts)
    hit = {v: _present(v, cand, cext) for v in expected}
    got = [v for v, h in hit.items() if h]
    miss = [v for v, h in hit.items() if not h]
    return len(got) / len(expected), (
        f"{len(got)} of {len(expected)} drawing dimensions the reference "
        f"exhibits are realised as a parameter or in the geometry"
        + (f"; missing {', '.join('%g' % v for v in miss)}" if miss else ""))


def score_all(cand_meas, ref_meas, cand_dir=None, ref_dir=None):
    cand, ref = Model(cand_meas, cand_dir), Model(ref_meas, ref_dir)
    cpts, rpts = cand.mesh_points(), ref.mesh_points()
    maps = [group_map(cand, ref, lvl) for lvl in ("groups", "bodies")]
    aligned = align(cpts, rpts) if cpts is not None and rpts is not None else None
    pose = None
    if aligned:
        res, _ = aligned
        kinds = ["proper"] if MIRROR_IS_DEFECT else list(res)
        kind = max(kinds, key=lambda k: res[k][0])
        pose = (res[kind][1], res[kind][2])
    best = lambda fn, *a: max((fn(cand, ref, m, *a) for m in maps),   # noqa: E731
                              key=lambda r: r[0])
    out = {}
    out["geometry matches ground truth"] = geometry(cand, ref, cpts, rpts, aligned)
    out["pantograph joints mated"] = best(joints_mated)
    out["mechanism articulates"] = best(articulates, pose)
    out["carries dimensional constraints"] = carries_constraints(cand, ref)
    out["dimensions drive the geometry"] = dims_drive(cand, ref)
    out["signature dimensions present"] = signature_dims(cand, ref, cpts, rpts)
    return out, cand
