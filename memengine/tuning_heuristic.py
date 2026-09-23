"""HeuristicTuner — the zero-config default tuner (telys-runtime, closed).

This is the engine-coupled tuner IMPLEMENTATION. It is part of the runtime, not the public SDK (D-30): it
reaches into the index internals to measure recall/latency and to decide IVF for oversized partitions. It
implements the public `telys.tuning.Tuner` interface and emits an auditable `telys.tuning.TuningPlan`. No
Algenta, no workload model — deterministic and embeddable.
"""
from __future__ import annotations

import time

from telys.tuning import Tuner, TuningPlan, _hash, _k, _workload_digest


class HeuristicTuner(Tuner):
    """Package the existing measured calibration as a plan: pick IVF for partitions above a row threshold,
    keep small ones exact, and report the recall the governed per-partition calibration will hold at apply."""
    name = "heuristic"

    def __init__(self, target_recall: float = 0.98, ivf_min_rows: int = 20000, sample: int = 200) -> None:
        self.target_recall = target_recall
        self.ivf_min_rows = ivf_min_rows
        self.sample = sample

    def tune_collection(self, collection, workload=None, objectives=None) -> TuningPlan:
        idx = collection.idx
        cons = (objectives or {}).get("constraints", {}) if objectives else {}
        target = float(cons.get("recall_at_10", cons.get("recall_at_k", self.target_recall)))
        pdir = getattr(idx, "pdir", None)
        if not pdir:                                 # never-built / never-sealed collection: nothing to tune
            return TuningPlan(
                collection=collection.name, tuner=self.name, target_recall=target,
                exact_crossover_rows=int(getattr(idx, "exact_crossover", 1 << 62)),
                ivf={"enabled": False, "min_rows": self.ivf_min_rows, "target_recall": target, "partitions": []},
                constraints=dict(cons), decision_trace=["no data yet (collection not built) — nothing to tune"],
                workload_hash=_hash(_workload_digest(workload)) if workload is not None else "",
                expected={"note": "empty collection; no measurement"})
        sizes = {k: v[1] for k, v in pdir.items()}
        big = sorted(((k, n) for k, n in sizes.items() if n >= self.ivf_min_rows), key=lambda x: -x[1])
        trace = [
            f"target recall floor = {target}",
            f"IVF for partitions >= {self.ivf_min_rows} rows: {len(big)} of {len(sizes)} "
            f"(exact contiguous scan kept for the rest)",
            "IVF nprobe is calibrated per-partition to the floor at apply; expected.* below is the CURRENT "
            "(pre-apply) exact-scan baseline, not the tuned IVF prediction",
        ]
        if not big:
            trace.append("no oversized partitions — pure exact slice scans; no IVF needed")
        plan = TuningPlan(
            collection=collection.name, tuner=self.name, target_recall=target,
            exact_crossover_rows=int(getattr(idx, "exact_crossover", 1 << 62)),
            ivf={"enabled": bool(big), "min_rows": self.ivf_min_rows, "target_recall": target,
                 "partitions": [{"key": _k(k), "rows": int(n)} for k, n in big[:64]]},
            constraints=dict(cons),
            decision_trace=trace,
            workload_hash=_hash(_workload_digest(workload)) if workload is not None else "",
        )
        plan.expected = self._measure(collection, target)
        return plan

    def _measure(self, collection, target):
        """Read-only CURRENT-state baseline: sample queries from the largest (worst-case) partition's own
        rows and measure recall@k vs an exact faiss scan. NOTE: this runs BEFORE apply builds the IVF, so the
        numbers describe the exact slice scan (the pre-tuning baseline), NOT the tuned IVF plan — keys are
        suffixed `_exact` and `recall_at_10_target` carries the floor the tuned plan will hold."""
        import numpy as np
        idx = collection.idx
        if not getattr(idx, "pdir", None) or idx.n == 0:
            return {}
        try:
            import faiss
        except Exception:  # noqa: BLE001
            return {"note": "faiss unavailable; expected metrics skipped", "recall_at_10_target": target}
        from telys.filters import Eq
        gk = max(idx.pdir, key=lambda k: idx.pdir[k][1])
        start, L = idx.pdir[gk]
        k = min(10, L)
        sub = np.ascontiguousarray(idx.base[start:start + L])
        rng = np.random.default_rng(0)
        a = rng.integers(0, L, size=min(self.sample, L, 256))
        q = sub[a] + 0.05 * rng.standard_normal((len(a), idx.dim)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        q = np.ascontiguousarray(q, np.float32)
        fl = faiss.IndexFlatIP(idx.dim); fl.add(sub); _, tp = fl.search(q, k)
        lat, hit = [], 0.0
        collection.search(q[0], k, where=Eq(idx.key_name, gk), target_recall=target)  # warm
        for i in range(len(q)):
            t = time.perf_counter_ns()
            r = collection.search(q[i], k, where=Eq(idx.key_name, gk), target_recall=target)
            lat.append(time.perf_counter_ns() - t)
            ext = r["ids"] if isinstance(r, dict) else r[0]
            got = {collection._id2int[x] for x in ext if x in collection._id2int}  # external -> internal logical
            truth = {int(idx.phys_rowid[start + p]) for p in tp[i] if p >= 0}       # faiss pads with -1 if k>L
            hit += len(got & truth) / max(1, len(truth))
        return {"basis": "exact_baseline_pre_apply",
                "p50_ms_exact": round(float(np.percentile(lat, 50)) / 1e6, 4),
                "p95_ms_exact": round(float(np.percentile(lat, 95)) / 1e6, 4),
                "recall_at_10_exact": round(hit / len(q), 4),
                "recall_at_10_target": target,
                "measured_on": f"partition '{_k(gk)}' ({L} rows)"}
