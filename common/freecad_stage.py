"""Base class for FreeCAD measurement stages run under freecadcmd.

Harness.run_freecad_stage launches a stage script with three env vars:
FC_STAGE_INPUT (the document to measure), FC_STAGE_OUT (a directory for
measure.json and any extra exports) and FC_STAGE_COMMON (the directory
holding common/, so the script can import this module). A stage subclasses
Stage, implements measure(doc) -> dict, and ends with `MyStage.run()`.

Stdlib only: this runs inside freecadcmd's own Python.
"""
import json
import os
import traceback


class Stage:
    """Subclass, implement measure(doc) -> dict, call run() at module end."""

    MEASURE_JSON = "measure.json"

    def __init__(self):
        self.doc_path = os.environ["FC_STAGE_INPUT"]
        self.out_dir = os.environ["FC_STAGE_OUT"]

    def open(self):
        """The FreeCAD document to measure; override for non-FCStd inputs."""
        import FreeCAD
        return FreeCAD.openDocument(self.doc_path)

    def measure(self, doc):
        """Return the JSON-serialisable measurement dict for `doc`."""
        raise NotImplementedError

    @classmethod
    def run(cls):
        stage = cls()
        result = {"input": stage.doc_path, "ok": False}
        try:
            result.update(stage.measure(stage.open()))
            result["ok"] = True
        except Exception:
            result["error"] = traceback.format_exc()[-2000:]
        os.makedirs(stage.out_dir, exist_ok=True)
        with open(os.path.join(stage.out_dir, cls.MEASURE_JSON), "w",
                  encoding="utf-8") as f:
            json.dump(result, f, indent=1)
        print("FC_STAGE_DONE ok=%s" % result["ok"])
        return 0 if result["ok"] else 1
