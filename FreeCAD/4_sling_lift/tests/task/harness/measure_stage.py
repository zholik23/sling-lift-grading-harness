"""Measurement stage for 4_sling_lift. Runs under freecadcmd, stdlib + FreeCAD.

Pulls out of the document everything the harness grades, as plain JSON,
and nothing that depends on how the candidate named or nested things:

  bodies     the rigid units the assembly is made of (the things joints
             reference). Per body: every solid's volume, area, centre of
             mass and inertia tensor in WORLD coordinates (the scorer
             builds rotation-, reflection- and modelling-order-invariant
             signatures from these); cylinder diameters, distances between
             parallel cylinder axes, local extents (for the drawing's
             signature dimensions); how much of its volume is dead
             imported geometry.
  joints     every object carrying a JointType, however it got there:
             type, active flag, which bodies its two references land on,
             both connector frames in world coordinates (with and without
             offsets), how far each connector sits from the material of
             the body it claims to be on, offset size, and limits.
  sketches   dimensional constraint census and an estimate of how
             constrained each sketch that feeds a solid actually is.
  carriers   every numeric value that could be a "key dimension":
             spreadsheet constants, driving sketch dimensions, feature
             lengths/angles. With whether it is named.
  probes     a sample of carriers nudged by 5%, the document recomputed,
             and whether the SOLIDS changed. A constraint that drives
             nothing is visible here and nowhere else.
  mesh       the assembled solids at the delivered pose, as a binary STL
             next to measure.json, for the scorer's pose comparison.

Every sub-measurement is fenced: a failure is recorded in `warnings` and
the rest still comes back. The probes run last because they mutate the
document; nothing measured before them can see their effect.
"""
import math
import os
import struct
import sys
import time

sys.path.insert(0, os.environ["FC_STAGE_COMMON"])

from common.freecad_stage import Stage                    # noqa: E402

PROBE_BUDGET_S = float(os.environ.get("SLING_PROBE_BUDGET_S", "240"))
PROBE_MAX = int(os.environ.get("SLING_PROBE_MAX", "16"))
MESH_DEFLECTION = float(os.environ.get("SLING_MESH_DEFLECTION", "0.6"))

ASSEMBLY_TYPES = ("Assembly::AssemblyObject",)
NON_BODY_TYPES = ("Assembly::JointGroup", "Assembly::ViewGroup",
                  "Assembly::BomGroup", "Assembly::SimulationGroup",
                  "Assembly::BomObject", "Spreadsheet::Sheet",
                  "App::Origin", "App::Line", "App::Plane", "App::Point")
FOLDER_TYPES = ("App::DocumentObjectGroup",)
CONTAINER_TYPES = ("PartDesign::Body", "App::Part") + ASSEMBLY_TYPES
#: Shapes with no history: imported or pasted geometry.
DEAD_TYPES = ("Part::Feature", "Mesh::Feature", "Part::FeatureExt",
              "PartDesign::FeatureBase")
DIM_CONSTRAINTS = ("Distance", "DistanceX", "DistanceY", "Radius",
                   "Diameter", "Angle")
#: Rough DoF a constraint removes, for the constrainedness estimate.
CONSTRAINT_DOF = {"Coincident": 2, "Horizontal": 1, "Vertical": 1,
                  "Parallel": 1, "Perpendicular": 1, "Tangent": 1,
                  "Equal": 1, "PointOnObject": 1, "Symmetric": 2,
                  "Distance": 1, "DistanceX": 1, "DistanceY": 1,
                  "Radius": 1, "Diameter": 1, "Angle": 1, "Lock": 2,
                  "Weight": 1, "SnellsLaw": 1, "InternalAlignment": 0,
                  "Block": 99}
GEO_DOF = {"LineSegment": 4, "Circle": 3, "ArcOfCircle": 5, "Point": 2,
           "Ellipse": 5, "ArcOfEllipse": 7, "BSplineCurve": 8,
           "ArcOfHyperbola": 7, "ArcOfParabola": 6}
LENGTH_PROP_TYPES = ("App::PropertyLength", "App::PropertyDistance",
                     "App::PropertyAngle")
SKIP_PROPS = ("Placement", "AttachmentOffset", "MapPathParameter",
              "Refine", "Offset2", "Tolerance")


def num(x):
    try:
        return float(getattr(x, "Value", x))
    except Exception:                                      # noqa: BLE001
        return None


def mat16(plc):
    return [float(v) for v in plc.toMatrix().A]


def vec(v):
    return [float(v.x), float(v.y), float(v.z)]


class SlingLiftStage(Stage):

    def measure(self, doc):
        import FreeCAD as App
        self.App = App
        self.doc = doc
        self.warn = []
        out = {"freecad_version": ".".join(App.Version()[:3])}

        t0 = time.time()
        try:
            doc.recompute()
        except Exception as exc:                           # noqa: BLE001
            self.warn.append(f"recompute raised: {exc}")
        out["recompute_s"] = round(time.time() - t0, 2)
        out["recompute_errors"] = [o.Name for o in doc.Objects
                                   if "Invalid" in getattr(o, "State", ())
                                   or "Error" in getattr(o, "State", ())]
        out["n_objects"] = len(doc.Objects)

        self.asm = self._top_assembly()
        out["assembly"] = self.asm is not None
        self.bodies = self._collect_bodies()
        out["bodies"] = [self._fence(self._measure_body, b, default={})
                         for b in self.bodies]
        self.body_index = {b["obj"].Name: i for i, b in enumerate(self.bodies)}
        self.alias = self._fence(self._source_map, default={})
        self._via = None
        out["subassemblies"] = getattr(self, "subasms", [])
        out["joints"], out["grounded"] = self._fence(
            self._measure_joints, default=([], []))
        out["sketches"] = self._fence(self._measure_sketches, default=[])
        self.carriers = self._fence(self._collect_carriers, default=[])
        out["carriers"] = [{k: v for k, v in c.items() if k != "_set"}
                           for c in self.carriers]
        out["mesh"] = self._fence(self._export_mesh, default=None)
        out["fingerprint"] = self._fingerprint()
        out["probes"], out["probe_budget_hit"] = self._fence(
            self._run_probes, default=([], False))
        out["warnings"] = self.warn[-60:]
        return out

    # ------------------------------------------------------------------
    def _fence(self, fn, *args, default=None):
        try:
            return fn(*args)
        except Exception as exc:                           # noqa: BLE001
            self.warn.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
            return default

    def _top_assembly(self):
        asms = [o for o in self.doc.Objects if o.TypeId in ASSEMBLY_TYPES]
        top = [a for a in asms
               if not any(p.TypeId in ASSEMBLY_TYPES for p in a.InListRecursive)]
        if not top:
            return None
        # several top-level assemblies: the one with the most joints
        return max(top, key=lambda a: sum(1 for o in a.OutListRecursive
                                          if hasattr(o, "JointType")))

    def _has_solids(self, shape):
        try:
            return shape is not None and not shape.isNull() and len(shape.Solids) > 0
        except Exception:                                  # noqa: BLE001
            return False

    def _collect_bodies(self):
        import Part
        bodies = []
        self.subasms = []
        if self.asm is not None:
            def children(container):
                kids = list(getattr(container, "Group", []) or [])
                if not kids:
                    tgt = getattr(container, "LinkedObject", None)
                    kids = list(getattr(tgt, "Group", []) or [])
                return kids

            def walk(container, path, sub=None):
                for ch in children(container):
                    if ch.TypeId in NON_BODY_TYPES or hasattr(ch, "JointType") \
                            or hasattr(ch, "ObjectToGround"):
                        continue
                    if ch.TypeId in FOLDER_TYPES:
                        walk(ch, path, sub)
                        continue
                    if self._is_subassembly(ch):
                        # a sub-assembly is a set of parts, not one part:
                        # flatten it, remembering whether it moves rigidly
                        info = {"name": ch.Name, "label": ch.Label,
                                "rigid": bool(getattr(ch, "Rigid", False)),
                                "type": ch.TypeId}
                        self.subasms.append(info)
                        walk(ch, path + ch.Name + ".", len(self.subasms) - 1)
                        continue
                    shape = None
                    for sub in (path + ch.Name + ".", ch.Name + "."):
                        try:
                            shape = Part.getShape(self.asm, sub,
                                                  needSubElement=False,
                                                  transform=True)
                        except Exception:                  # noqa: BLE001
                            shape = None
                        if self._has_solids(shape):
                            break
                    if not self._has_solids(shape):
                        try:
                            shape = Part.getShape(ch)
                            shape = shape.transformGeometry(
                                self.asm.Placement.toMatrix()) \
                                if not self.asm.Placement.isIdentity() else shape
                        except Exception:                  # noqa: BLE001
                            shape = None
                    if self._has_solids(shape):
                        bodies.append({"obj": ch, "shape": shape, "sub": sub})
            walk(self.asm, "")
            if bodies:
                return bodies
            self.warn.append("assembly holds no solid bodies; using document")
        # No assembly: every final (unconsumed) solid-bearing object.
        cands = []
        for o in self.doc.Objects:
            shape = getattr(o, "Shape", None)
            if not self._has_solids(shape) or o.TypeId in ASSEMBLY_TYPES:
                continue
            parent = o.getParentGeoFeatureGroup() if hasattr(
                o, "getParentGeoFeatureGroup") else None
            if parent is not None and parent.TypeId == "PartDesign::Body":
                continue
            consumed = any(p.TypeId not in CONTAINER_TYPES + FOLDER_TYPES
                           and self._has_solids(getattr(p, "Shape", None))
                           for p in o.InList)
            if not consumed:
                cands.append(o)
        visible = [o for o in cands if getattr(o, "Visibility", True)]
        for o in (visible or cands):
            try:
                bodies.append({"obj": o, "shape": Part.getShape(o)})
            except Exception:                              # noqa: BLE001
                bodies.append({"obj": o, "shape": o.Shape})
        return bodies

    def _is_subassembly(self, obj):
        if obj.TypeId in ("Assembly::AssemblyLink",) + ASSEMBLY_TYPES:
            return True
        tgt = getattr(obj, "LinkedObject", None)
        return obj.TypeId == "App::Link" and tgt is not None \
            and tgt.TypeId in ASSEMBLY_TYPES

    def _source_map(self):
        """Joints inside a sub-assembly reference the sub-assembly's own
        components, not the top-level copies the AssemblyLink shows. Map
        each such component to the body that stands for it (only when the
        mapping is unique)."""
        alias, clash = {}, set()
        for i, b in enumerate(self.bodies):
            if b.get("sub") is None:
                continue
            src = getattr(b["obj"], "LinkedObject", None)
            if src is None or src.Name in self.body_index:
                continue
            if src.Name in alias:
                clash.add(src.Name)
            alias[src.Name] = i
        for n in clash:
            alias.pop(n, None)
        return alias

    def _frame_map(self, body_i, src):
        """Rigid map from the frame a sub-assembly joint is expressed in
        (its component's global placement) to the top-level frame, found
        by matching the same shape in both. None when it cannot be
        established, which leaves that joint's frames as they are."""
        import FreeCAD as App
        import Part
        b = self.bodies[body_i]
        if "_fmap" in b:
            return b["_fmap"]
        M = None
        try:
            cs = Part.getShape(src, "", needSubElement=False, transform=False)
            cs = cs.copy()
            if hasattr(src, "getGlobalPlacement"):
                gp = src.getGlobalPlacement()
            else:
                gp = getattr(src, "Placement", App.Placement())
            cs.transformShape(gp.toMatrix(), True)
            a = [v.Point for v in cs.Vertexes]
            c = [v.Point for v in b["shape"].Vertexes]
            if len(a) == len(c) and len(a) >= 3:
                fa, fc = self._frame3(a), self._frame3(c)
                if fa is not None and fc is not None:
                    M = fc.multiply(fa.inverse())
                    res = max((M.multVec(p) - q).Length
                              for p, q in list(zip(a, c))[:200])
                    if res > max(0.05, 1e-4 * b["shape"].BoundBox.DiagonalLength):
                        self.warn.append(f"frame map for {b['obj'].Name}: residual {res:.3f}")
                        M = None
        except Exception as exc:                           # noqa: BLE001
            self.warn.append(f"frame map for {b['obj'].Name}: {exc}")
        b["_fmap"] = M
        _ = App
        return M

    @staticmethod
    def _frame3(pts):
        """An orthonormal frame (as a Matrix) built from three
        well-spread points of a point list -- the same three indices are
        used for both copies, so corresponding frames correspond."""
        import FreeCAD as App
        p0 = pts[0]
        i1 = max(range(len(pts)), key=lambda k: (pts[k] - p0).Length)
        e1 = pts[i1] - p0
        if e1.Length < 1e-6:
            return None
        e1 = e1.normalize()
        def off(k):
            d = pts[k] - p0
            return (d - e1 * d.dot(e1)).Length
        i2 = max(range(len(pts)), key=off)
        if off(i2) < 1e-6:
            return None
        d = pts[i2] - p0
        e2 = (d - e1 * d.dot(e1)).normalize()
        e3 = e1.cross(e2)
        return App.Matrix(e1.x, e2.x, e3.x, p0.x,
                          e1.y, e2.y, e3.y, p0.y,
                          e1.z, e2.z, e3.z, p0.z,
                          0, 0, 0, 1)

    def _dead_volume(self, obj):
        """Volume of this body that is history-free imported geometry."""
        tgt = obj
        for _ in range(6):                       # follow App::Link chains
            nxt = getattr(tgt, "LinkedObject", None)
            if nxt is None or nxt is tgt:
                break
            tgt = nxt
        if tgt.TypeId in DEAD_TYPES:
            return float(tgt.Shape.Volume)
        dead = 0.0
        if tgt.TypeId == "PartDesign::Body":
            base = getattr(tgt, "BaseFeature", None)
            if base is not None and self._has_solids(getattr(base, "Shape", None)):
                dead += float(base.Shape.Volume)
            for f in getattr(tgt, "Group", []):
                if f.TypeId == "PartDesign::FeatureBase":
                    b = getattr(f, "BaseFeature", None)
                    if b is not None and self._has_solids(getattr(b, "Shape", None)):
                        dead += float(b.Shape.Volume)
        return dead

    def _measure_body(self, b):
        import Part
        obj, shape = b["obj"], b["shape"]
        solids = []
        for s in shape.Solids[:200]:
            try:
                m = s.MatrixOfInertia.A
                solids.append({"V": float(s.Volume), "A": float(s.Area),
                               "com": vec(s.CenterOfMass),
                               "I": [m[0], m[1], m[2], m[4], m[5], m[6],
                                     m[8], m[9], m[10]]})
            except Exception as exc:                       # noqa: BLE001
                self.warn.append(f"solid in {obj.Name}: {exc}")
        cyl_d, axes = set(), []
        for f in shape.Faces[:600]:
            surf = f.Surface
            if isinstance(surf, Part.Cylinder):
                cyl_d.add(round(2 * surf.Radius, 2))
                axes.append((surf.Center, surf.Axis))
        axis_d = set()
        for i in range(min(len(axes), 80)):
            for j in range(i + 1, min(len(axes), 80)):
                (c1, a1), (c2, a2) = axes[i], axes[j]
                if abs(abs(a1.dot(a2)) - 1.0) > 1e-4:
                    continue
                d = (c2 - c1)
                d = d - a1 * d.dot(a1)
                if d.Length > 0.5:
                    axis_d.add(round(d.Length, 2))
        local = shape.copy()
        try:
            gp = obj.getGlobalPlacement() if hasattr(obj, "getGlobalPlacement") \
                else obj.Placement
            local.transformShape(gp.inverse().toMatrix())
        except Exception:                                  # noqa: BLE001
            pass
        bb = local.BoundBox
        gbb = shape.BoundBox
        linked = getattr(obj, "LinkedObject", None)
        return {"name": obj.Name, "label": obj.Label, "type": obj.TypeId,
                "linked_type": getattr(linked, "TypeId", None),
                "sub": b.get("sub"),
                "n_children": len(getattr(obj, "Group", []) or []),
                "solids": solids, "cyl_d": sorted(cyl_d)[:300],
                "axis_d": sorted(axis_d)[:400],
                "extent": [bb.XLength, bb.YLength, bb.ZLength],
                "diag": float(gbb.DiagonalLength),
                "volume": float(shape.Volume),
                "dead_volume": self._fence(self._dead_volume, obj, default=0.0)}

    # ------------------------------------------------------------------
    def _resolve_body(self, ref):
        """Which body a joint reference lands on: walk the subname path."""
        if ref is None:
            return None, None
        try:
            obj, subs = ref[0], ref[1]
        except Exception:                                  # noqa: BLE001
            return None, None
        if obj is None:
            return None, None
        if isinstance(subs, str):
            subs = [subs]
        subs = list(subs or [""])
        self._via = None
        if obj.Name in self.body_index:
            return self.body_index[obj.Name], subs
        if obj.Name in self.alias:
            self._via = obj
            return self.alias[obj.Name], subs
        for sub in subs:
            for tok in (sub or "").split(".")[:-1]:
                if tok in self.body_index:
                    return self.body_index[tok], subs
                if tok in self.alias:
                    self._via = self.doc.getObject(tok)
                    return self.alias[tok], subs
        # reference rooted at an object nested inside a body
        for p in getattr(obj, "InListRecursive", []):
            if p.Name in self.body_index:
                return self.body_index[p.Name], subs
            if p.Name in self.alias:
                self._via = p
                return self.alias[p.Name], subs
        return None, subs

    def _global_jcs(self, plc, ref, body_i):
        try:
            import UtilsAssembly
            return UtilsAssembly.getJcsGlobalPlc(plc, ref)
        except Exception:                                  # noqa: BLE001
            pass
        if body_i is None:
            return plc
        obj = self.bodies[body_i]["obj"]
        gp = obj.getGlobalPlacement() if hasattr(obj, "getGlobalPlacement") \
            else obj.Placement
        return gp.multiply(plc)

    def _axes(self, body_i):
        """Cylinder axes of a body (holes and pins), cached: a pivot's
        connector sits on one of these, not on the material itself."""
        import Part
        b = self.bodies[body_i]
        if "_axes" not in b:
            axes = []
            for f in b["shape"].Faces[:800]:
                srf = f.Surface
                if isinstance(srf, Part.Cylinder):
                    try:
                        v0, v1 = f.ParameterRange[2:4]
                    except Exception:                      # noqa: BLE001
                        v0, v1 = -1e9, 1e9
                    axes.append((srf.Center, srf.Axis, float(srf.Radius),
                                 float(min(v0, v1)), float(max(v0, v1))))
            b["_axes"] = axes
        return b["_axes"]

    def _anchor(self, plc, body_i):
        """How far a connector sits from the body it claims: the smaller
        of its distance to the material and its distance to a hole/pin
        axis of that body (within the cylinder's length, plus a radius)."""
        import Part
        if body_i is None:
            return None
        p = plc.Base
        best = None
        try:
            v = Part.Vertex(p)
            best = float(self.bodies[body_i]["shape"].distToShape(v)[0])
        except Exception:                                  # noqa: BLE001
            pass
        try:
            for c, a, r, v0, v1 in self._axes(body_i):
                d = p - c
                t = d.dot(a)
                if t < v0 - r - 1.0 or t > v1 + r + 1.0:
                    continue
                perp = (d - a * t).Length
                best = perp if best is None else min(best, perp)
        except Exception as exc:                           # noqa: BLE001
            self.warn.append(f"axis anchor: {exc}")
        return best

    def _measure_joints(self):
        App = self.App
        joints, grounded = [], []
        scope = self.asm.OutListRecursive if self.asm is not None else self.doc.Objects
        for o in scope:
            if hasattr(o, "ObjectToGround"):
                g = getattr(o, "ObjectToGround", None)
                if g is not None and g.Name in self.body_index:
                    grounded.append(self.body_index[g.Name])
                continue
            if not hasattr(o, "JointType"):
                continue
            j = {"name": o.Name, "type": str(o.JointType),
                 "active": bool(getattr(o, "Activated", True)),
                 "suppressed": bool(getattr(o, "Suppressed", False))}
            refs = []
            for k in ("1", "2"):
                ref = getattr(o, "Reference" + k, None)
                self._via = None
                bi, subs = self._resolve_body(ref)
                plc = getattr(o, "Placement" + k, App.Placement())
                off = getattr(o, "Offset" + k, None)
                if not isinstance(off, App.Placement):
                    off = App.Placement()
                via = self._via
                g = self._global_jcs(plc, ref, bi)
                go = g.multiply(off)
                if via is not None and bi is not None:
                    M = self._frame_map(bi, via)
                    if M is not None:
                        Mp = App.Placement(M)
                        g, go = Mp.multiply(g), Mp.multiply(go)
                    else:
                        # a frame we cannot place in the top-level
                        # assembly cannot be graded; leave it unresolved
                        # rather than mix coordinate frames
                        bi = None
                    j["mapped%s" % k] = M is not None
                j["b" + k] = bi
                try:
                    j["ref%s_raw" % k] = [ref[0].Name if ref and ref[0] else None,
                                         list(subs or [])[:2]]
                except Exception:                          # noqa: BLE001
                    j["ref%s_raw" % k] = None
                j["g" + k] = mat16(g)
                j["g%so" % k] = mat16(go)
                j["anchor" + k] = self._anchor(g, bi)
                j["offset%s_mag" % k] = float(off.Base.Length)
                j["offset%s_rot" % k] = float(math.degrees(off.Rotation.Angle))
                refs.append(subs)
            legacy = getattr(o, "Offset", None)
            if isinstance(legacy, App.Placement):
                j["offset1_mag"] = max(j["offset1_mag"], float(legacy.Base.Length))
            j["distance"] = num(getattr(o, "Distance", 0.0))
            lim = {}
            for key, en in (("AngleMin", "EnableAngleMin"),
                            ("AngleMax", "EnableAngleMax"),
                            ("LengthMin", "EnableLengthMin"),
                            ("LengthMax", "EnableLengthMax")):
                if getattr(o, en, False):
                    lim[key] = num(getattr(o, key, None))
            j["limits"] = lim
            j["n_refs"] = sum(1 for r in refs if r)
            joints.append(j)
        return joints, sorted(set(grounded))

    # ------------------------------------------------------------------
    def _feeds_solid(self, obj):
        for p in obj.InListRecursive:
            if p.TypeId in CONTAINER_TYPES + FOLDER_TYPES:
                continue
            if self._has_solids(getattr(p, "Shape", None)):
                return True
        return False

    def _expr_paths(self, obj):
        try:
            return [p for p, _ in obj.ExpressionEngine]
        except Exception:                                  # noqa: BLE001
            return []

    def _measure_sketches(self):
        out = []
        for sk in self.doc.Objects:
            if sk.TypeId != "Sketcher::SketchObject":
                continue
            geo_dof = 0
            for g in sk.Geometry:
                geo_dof += GEO_DOF.get(type(g).__name__, 4)
            removed, n_dims, n_named = 0, 0, 0
            for c in sk.Constraints:
                if c.Type in DIM_CONSTRAINTS and not getattr(c, "Driving", True):
                    continue
                removed += CONSTRAINT_DOF.get(c.Type, 1)
                if c.Type in DIM_CONSTRAINTS:
                    n_dims += 1
                    n_named += bool(c.Name)
            removed += 2 * len(getattr(sk, "ExternalGeometry", []) or [])
            fully = getattr(sk, "FullyConstrained", None)
            out.append({"name": sk.Name, "used": self._feeds_solid(sk),
                        "geo_dof": geo_dof, "removed_dof": removed,
                        "fully": None if fully is None else bool(fully),
                        "n_dims": n_dims, "n_named": n_named})
        return out

    def _collect_carriers(self):
        App = self.App
        cars = []
        for o in self.doc.Objects:
            if o.TypeId == "Spreadsheet::Sheet":
                cars += self._sheet_carriers(o)
        for sk in self.doc.Objects:
            if sk.TypeId != "Sketcher::SketchObject" or not self._feeds_solid(sk):
                continue
            exprs = self._expr_paths(sk)
            for i, c in enumerate(sk.Constraints):
                if c.Type not in DIM_CONSTRAINTS or not getattr(c, "Driving", True):
                    continue
                if any((".Constraints." + c.Name) == p or
                       p.endswith("Constraints[%d]" % i) or
                       (c.Name and p.endswith("Constraints." + c.Name))
                       for p in exprs):
                    continue                    # derived; its source is probed
                ang = c.Type == "Angle"
                val = math.degrees(c.Value) if ang else c.Value
                if c.Type == "Radius":
                    extra = [2 * val]
                else:
                    extra = []
                cars.append({"kind": "sketch", "owner": sk.Name, "key": i,
                             "value": float(val), "alt": extra,
                             "named": bool(c.Name),
                             "unit": "deg" if ang else "mm",
                             "_set": ("sketch", sk, i, ang)})
        for o in self.doc.Objects:
            if o.TypeId.startswith("Sketcher") or o.TypeId == "Spreadsheet::Sheet":
                continue
            if not (o.TypeId.startswith("PartDesign::") or o.TypeId.startswith("Part::")):
                continue
            if o.TypeId in DEAD_TYPES:
                continue
            exprs = set(self._expr_paths(o))
            for p in o.PropertiesList:
                if p in SKIP_PROPS or p in exprs or ("." + p) in exprs:
                    continue
                try:
                    tp = o.getTypeIdOfProperty(p)
                except Exception:                          # noqa: BLE001
                    continue
                if tp not in LENGTH_PROP_TYPES:
                    continue
                if "Hidden" in o.getPropertyStatus(p) or \
                        "ReadOnly" in o.getPropertyStatus(p):
                    continue
                v = num(getattr(o, p))
                if v is None or abs(v) < 1e-9:
                    continue
                cars.append({"kind": "feature", "owner": o.Name, "key": p,
                             "value": v, "alt": [], "named": False,
                             "unit": "deg" if tp == "App::PropertyAngle" else "mm",
                             "_set": ("prop", o, p, None)})
        _ = App
        return cars

    def _sheet_carriers(self, sh):
        cells = []
        try:
            cells = list(sh.getUsedCells())
        except Exception:                                  # noqa: BLE001
            for p in sh.PropertiesList:
                try:
                    c = sh.getCellFromAlias(p)
                except Exception:                          # noqa: BLE001
                    c = None
                if c:
                    cells.append(c)
        out = []
        for c in cells:
            try:
                content = sh.getContents(c)
            except Exception:                              # noqa: BLE001
                continue
            if not content or content.startswith("="):
                continue                         # formulas are derived
            v = num(sh.get(c))
            if v is None:
                continue
            try:
                alias = sh.getAlias(c)
            except Exception:                              # noqa: BLE001
                alias = None
            out.append({"kind": "sheet", "owner": sh.Name, "key": c,
                        "value": v, "alt": [], "named": bool(alias),
                        "unit": "mm", "alias": alias,
                        "_set": ("sheet", sh, c, content)})
        return out

    # ------------------------------------------------------------------
    def _export_mesh(self):
        pts_all, tris_all = [], []
        for b in self.bodies:
            pts, tris = b["shape"].tessellate(MESH_DEFLECTION)
            base = len(pts_all)
            pts_all += [(p.x, p.y, p.z) for p in pts]
            tris_all += [(a + base, c + base, d + base) for a, c, d in tris]
        path = os.path.join(self.out_dir, "assembly.stl")
        os.makedirs(self.out_dir, exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"sling-lift harness mesh".ljust(80, b" "))
            f.write(struct.pack("<I", len(tris_all)))
            for a, c, d in tris_all:
                f.write(struct.pack("<12fH", 0, 0, 0, *pts_all[a], *pts_all[c],
                                    *pts_all[d], 0))
        return {"file": "assembly.stl", "triangles": len(tris_all)}

    def _final_solids(self):
        """What a probe compares: each body's solids, re-read fresh, by
        quantities that do not depend on where the body is. A recompute
        re-solves the assembly and moves parts slightly; that must not
        read as 'the dimension changed the geometry'."""
        import Part
        sig = []
        for b in self.bodies:
            o = b["obj"]
            try:
                s = Part.getShape(o)
                sig.append([float(s.Volume), float(s.Area)])
            except Exception:                              # noqa: BLE001
                sig.append(None)
        return sig

    def _fingerprint(self):
        return self._final_solids()

    @staticmethod
    def _differs(a, b):
        if a is None or b is None or len(a) != len(b):
            return True
        for x, y in zip(a, b):
            if (x is None) != (y is None):
                return True
            if x is None:
                continue
            for u, v in zip(x, y):
                if abs(u - v) > max(1e-4, 1e-6 * abs(v)):
                    return True
        return False

    def _set(self, car, value):
        kind, owner, key, extra = car["_set"]
        App = self.App
        if kind == "sheet":
            owner.set(key, repr(float(value)))
        elif kind == "sketch":
            q = App.Units.Quantity("%r deg" % value) if extra \
                else App.Units.Quantity("%r mm" % value)
            owner.setDatum(key, q)
        else:
            setattr(owner, key, value)

    def _restore(self, car):
        kind, owner, key, extra = car["_set"]
        if kind == "sheet":
            owner.set(key, extra)
        else:
            self._set(car, car["value"])

    def _pick_probes(self):
        """Named carriers first (they are what 'key dimensions' means),
        then the rest, sampled evenly so one big sketch cannot hog it."""
        named = [i for i, c in enumerate(self.carriers) if c["named"]]
        rest = [i for i, c in enumerate(self.carriers) if not c["named"]]

        def spread(ix, n):
            if len(ix) <= n:
                return ix
            step = len(ix) / float(n)
            return [ix[int(k * step)] for k in range(n)]
        n_named = min(len(named), max(PROBE_MAX // 2, PROBE_MAX - len(rest)))
        chosen = spread(named, n_named)
        return chosen + spread(rest, PROBE_MAX - len(chosen))

    def _run_probes(self):
        base = self._final_solids()
        results, t0, hit = [], time.time(), False
        for i in self._pick_probes():
            if time.time() - t0 > PROBE_BUDGET_S:
                hit = True
                break
            car = self.carriers[i]
            v = car["value"]
            res = {"carrier": i, "changed": 0.0}
            for factor in (1.05, 0.95):
                nv = v * factor
                try:
                    self._set(car, nv)
                    self.doc.recompute()
                    errs = [o.Name for o in self.doc.Objects
                            if "Invalid" in getattr(o, "State", ())]
                    after = self._final_solids()
                    if errs:
                        res["changed"] = max(res["changed"], 0.5)
                        res["detail"] = "recompute errors: %s" % errs[:3]
                        continue
                    if self._differs(base, after):
                        res["changed"] = 1.0
                        break
                except Exception as exc:                   # noqa: BLE001
                    res["detail"] = "%s: %s" % (type(exc).__name__, exc)
                finally:
                    try:
                        self._restore(car)
                        self.doc.recompute()
                    except Exception:                      # noqa: BLE001
                        pass
            if self._differs(base, self._final_solids()):
                res["restore_failed"] = True
                self.warn.append("probe %d did not restore; stopping" % i)
                results.append(res)
                break
            results.append(res)
        return results, hit


SlingLiftStage.run()
