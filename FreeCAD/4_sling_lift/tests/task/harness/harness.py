#!/usr/bin/env python3
"""Grading harness for 4_sling_lift.

    python3 tests/task/harness/harness.py solution/solution.FCStd

Measures the candidate and the reference with the same stage
(measure_stage.py, under freecadcmd, out of process) and scores the
candidate against the reference with scoring.py. Nothing here imports
FreeCAD.

COMPONENTS (weights unchanged from the stub; max_score stays 6)

    executes and builds geometry      gate: the stage ran and produced solids
    geometry matches ground truth     the part set by mass signature,
                                      pose-free and reflection-tolerant
    pantograph joints mated           reference pivots realised by honest
                                      joints between the matching bodies
    mechanism articulates             jointed mechanism volume x Jacobian
                                      mobility x (0.5 + 0.5 x joint limits,
                                      incl. can the stroke reach the drawn
                                      configuration)
    carries dimensional constraints   live (history-bearing) volume x sketch
                                      constrainedness, relative to reference
    dimensions drive the geometry     5% perturbation probes: did the solids
                                      move? relative to reference
    signature dimensions present      drawing dimensions realised as a
                                      parameter or in the geometry

See NOTES.md beside this file for what is measured, what is let go, and
the mirror decision.

The reference is measured once and cached next to it, keyed on the
reference file's and the stage script's hashes. For testing without
FreeCAD, a directory holding a measure.json (and assembly.stl) can be
passed instead of an .FCStd, for the candidate or -- via SLING_REFERENCE
-- for the reference.
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _repo_root(start):
    d = start
    for _ in range(8):
        if (d / "common" / "__init__.py").is_file():
            return d
        if d.parent == d:
            break
        d = d.parent
    return start


REPO = _repo_root(HERE)
sys.path[:0] = [str(HERE), str(REPO), "/opt"]

from common.harness_base import Harness                   # noqa: E402

ALL_CRITERIA = {
    "executes and builds geometry": 0,
    "geometry matches ground truth": 1,
    "pantograph joints mated": 1,
    "mechanism articulates": 1,
    "carries dimensional constraints": 1,
    "dimensions drive the geometry": 1,
    "signature dimensions present": 1,
}
GATE = "executes and builds geometry"


def _sha(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


class SlingLiftHarness(Harness):
    MUST_PASS = (GATE,)
    WEIGHTS = ALL_CRITERIA
    BUILD_TIMEOUT_S = 600

    # -- measurement ----------------------------------------------------
    def _measure(self, path, keep_dir):
        """(measurement, directory holding its mesh, error)."""
        p = Path(path)
        if p.is_dir() and (p / "measure.json").is_file():   # recorded
            return json.loads((p / "measure.json").read_text()), p, None
        try:
            meas, err = self.run_freecad_stage(str(p), out_dir=str(keep_dir))
        except SystemExit as exc:
            return None, None, f"setup: {exc}"
        except Exception as exc:                            # noqa: BLE001
            return None, None, f"{type(exc).__name__}: {exc}"
        return meas, Path(keep_dir), err

    def _reference_path(self):
        env = os.environ.get("SLING_REFERENCE")
        if env:
            return Path(env)
        for p in (HERE.parent / "reference" / "solution.FCStd",
                  self.task_dir() / "solution" / "solution.FCStd"):
            if p.is_file():
                return p
        return HERE.parent / "reference" / "solution.FCStd"

    def _reference(self):
        ref = self._reference_path()
        if ref.is_dir():
            return self._measure(ref, ref)
        if not ref.is_file():
            return None, None, f"reference not found at {ref} (run tools/fetch.py)"
        key = _sha(ref) + "-" + _sha(self.freecad_stage(), 8)
        for base in (HERE.parent / "reference" / ".cache",
                     Path(tempfile.gettempdir()) / "sling_lift_ref_cache"):
            cache = base / key
            if (cache / "measure.json").is_file():
                return json.loads((cache / "measure.json").read_text()), cache, None
            try:
                cache.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            meas, d, err = self._measure(ref, cache)
            if meas is not None:
                (cache / "measure.json").write_text(json.dumps(meas))
            else:
                shutil.rmtree(cache, ignore_errors=True)
            return meas, d, err
        return None, None, "no writable cache directory for the reference"

    def build_state(self, candidate_path):
        keep = os.environ.get("SLING_KEEP_DIR")       # debugging: keep it
        if keep:
            work = Path(keep) / Path(candidate_path).stem
            work.mkdir(parents=True, exist_ok=True)
        else:
            work = Path(tempfile.mkdtemp(prefix="sling_cand_"))
        cand, cdir, cerr = self._measure(candidate_path, work)
        ref, rdir, rerr = self._reference()
        return {"candidate": candidate_path, "measurement": cand or {},
                "cand_dir": cdir, "error": cerr,
                "reference": ref, "ref_dir": rdir, "ref_error": rerr,
                "work": work}

    # -- scoring --------------------------------------------------------
    def checks(self, state):
        out = {}
        meas, err = state["measurement"], state["error"]
        vol = sum(b.get("volume", 0.0) for b in meas.get("bodies", []) if b)
        if err is not None:
            out[GATE] = (GATE, 0.0, f"the measurement stage failed: {err}")
        elif vol <= 0:
            out[GATE] = (GATE, 0.0, "the document opened but holds no solid geometry")
        else:
            nerr = len(meas.get("recompute_errors", []))
            out[GATE] = (GATE, 1.0,
                         f"rebuilt: {len(meas.get('bodies', []))} bodies, "
                         f"{vol:.0f} mm^3 of solid"
                         + (f" ({nerr} features report errors)" if nerr else ""))
        if out[GATE][1] < 1.0 or state["reference"] is None:
            why = (out[GATE][2] if out[GATE][1] < 1.0 else
                   f"cannot grade without the reference: {state['ref_error']}")
            for name in ALL_CRITERIA:
                out.setdefault(name, (name, 0.0, why))
            self._cleanup(state)
            return out
        import scoring
        try:
            comps, _ = scoring.score_all(meas, state["reference"],
                                         state["cand_dir"], state["ref_dir"])
        finally:
            self._cleanup(state)
        for name, (score, desc) in comps.items():
            out[name] = (name, round(float(score), 4), desc)
        return out

    def main(self):
        """The base envelope, plus the sentence behind every number.

        finalize() keeps only the numbers; the descriptions are what tell
        a reader WHY a component scored what it did, so they ride along
        under `details` (test.sh reads only score/max_score/passed)."""
        from common.harness_base import (candidate_from_argv, finalize,
                                         timeout_from_argv)
        candidate = candidate_from_argv()
        self.timeout = timeout_from_argv(self.BUILD_TIMEOUT_S)
        checks = self.checks(self.build_state(candidate))
        env = finalize(self.task_dir(), checks, must_pass=self.MUST_PASS,
                       weights=self.WEIGHTS)
        env["details"] = {k: v[2] for k, v in checks.items()}
        return env

    @staticmethod
    def _cleanup(state):
        if not os.environ.get("SLING_KEEP_DIR"):
            shutil.rmtree(state.get("work"), ignore_errors=True)


main = SlingLiftHarness.as_main()

if __name__ == "__main__":
    SlingLiftHarness.cli()
