#!/usr/bin/env python3
"""Re-score saved real measurements without FreeCAD.

    python3 tests/task/harness/rescore.py <measure_root>

<measure_root> holds one folder per model (measure.json + assembly.stl), as
written when the harness runs with SLING_KEEP_DIR set; `solution/` is used
as the reference."""
import json, os, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
root = Path(sys.argv[1])
env = dict(os.environ, SLING_REFERENCE=str(root / "solution"))
env.pop("SLING_KEEP_DIR", None)
order = ["solution"] + sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "solution")
names = None
for n in order:
    r = subprocess.run([sys.executable, str(HERE / "harness.py"), str(root / n)],
                       env=env, capture_output=True, text=True)
    e = json.loads(r.stdout)
    if names is None:
        names = list(e["subscores"])
        print(f"{'model':<52} {'score':>6}  " + " ".join(f"{s[:5]:>5}" for s in
              ["gate", "geom", "mated", "artic", "cons", "drive", "sig"]))
    print(f"{n.replace('adversarial_', '')[:52]:<52} {e['score']:>6.2f}  " +
          " ".join(f"{e['subscores'][k]:>5.2f}" for k in names))
    if "-v" in sys.argv:
        for k, v in e["details"].items():
            print("    *", k, ":", v[:400])
