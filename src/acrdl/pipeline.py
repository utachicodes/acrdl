"""End-to-end review-and-repair pipeline.

problem + buggy program
  -> generate N candidate patches
  -> keep plausible ones (pass every visible test)
  -> build inputs (visible + fuzzed) and a differential behaviour matrix
  -> behavioural clustering + static smells (+ optional symbolic check)
  -> evidence vectors
  -> verdict per patch: trained fusion model if provided, else the transparent
     disagreement rule (acrdl.baselines.disagreement.HeuristicPredictor).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional, Protocol

from acrdl.baselines.disagreement import HeuristicPredictor
from acrdl.config import Config
from acrdl.evidence import differential as D
from acrdl.evidence.clustering import Clustering, cluster
from acrdl.evidence.features import build_evidence
from acrdl.evidence.inputs import build_inputs
from acrdl.evidence.static import StaticFeatures, static_features
from acrdl.evidence.symbolic import SymbolicVerdict, diff_behavior
from acrdl.evidence.testing import describe_failures, run_tests
from acrdl.generation.generator import PatchGenerator
from acrdl.types import BuggyProgram, Evidence, Label, Patch


class Predictor(Protocol):
    def predict(self, patches: list[Patch], evidence: Mapping[str, Evidence]
                ) -> dict[str, tuple[Label, list[float]]]: ...


@dataclass
class PatchReport:
    patch: Patch
    plausible: bool
    fail_to_pass: bool
    evidence: Optional[Evidence] = None
    static: Optional[StaticFeatures] = None
    verdict: Optional[Label] = None
    probs: list[float] = field(default_factory=list)
    reason: str = ""


@dataclass
class BugReport:
    bug: BuggyProgram
    patches: list[PatchReport]
    n_inputs: int = 0
    clustering: Optional[Clustering] = None
    matrix: Optional[D.BehaviourMatrix] = None
    ambiguity_witnesses: list[tuple[tuple, dict[str, list[str]]]] = field(default_factory=list)


class Pipeline:
    def __init__(self, config: Config, generator: PatchGenerator,
                 predictor: Optional[Predictor] = None):
        self.cfg = config
        self.generator = generator
        self.predictor = predictor or HeuristicPredictor()

    def run(self, bug: BuggyProgram) -> BugReport:
        ec = self.cfg.evidence
        prob = bug.problem
        entry = prob.entry_point
        visible = prob.visible_tests
        buggy_pass = run_tests(bug.source, entry, visible, ec.exec_timeout_s, ec.memory_limit_mb)

        n_failing = sum(not b for b in buggy_pass)
        reports: list[PatchReport] = []
        fixed: dict[str, float] = {}
        failing = describe_failures(bug.source, entry, visible, ec.exec_timeout_s,
                                    ec.memory_limit_mb) if n_failing else []
        for patch in self.generator.generate(bug, self.cfg.generator.n_candidates, failing):
            passed = run_tests(patch.source, entry, visible, ec.exec_timeout_s,
                               ec.memory_limit_mb)
            patch.plausible = all(passed)
            n_fixed = sum(p and not b for p, b in zip(passed, buggy_pass))
            fixed[patch.patch_id] = n_fixed / n_failing if n_failing else 1.0
            reports.append(PatchReport(patch, patch.plausible, n_fixed > 0))

        plausible = [r for r in reports if r.plausible]
        result = BugReport(bug, reports)
        if not plausible:
            return result

        inputs = build_inputs([t.args for t in visible], ec.n_fuzz_inputs, seed=self.cfg.seed)
        sources = {r.patch.patch_id: r.patch.source for r in plausible}
        matrix = D.build_matrix(bug.source, entry, sources, inputs, ec.exec_timeout_s,
                                ec.memory_limit_mb)
        clustering = cluster(matrix)
        static = {r.patch.patch_id: static_features(bug.source, r.patch.source, visible)
                  for r in plausible}
        symbolic: dict[str, SymbolicVerdict] = {}
        if ec.use_symbolic:
            symbolic = {pid: diff_behavior(bug.source, src, entry, ec.symbolic_timeout_s)
                        for pid, src in sources.items()}
        evidence = build_evidence(matrix, clustering, static,
                                  {r.patch.patch_id: r.fail_to_pass for r in plausible},
                                  symbolic, visible_fixed=fixed, n_generated=len(reports))
        verdicts = self.predictor.predict([r.patch for r in plausible], evidence)
        for r in plausible:
            pid = r.patch.patch_id
            r.evidence, r.static = evidence[pid], static[pid]
            r.verdict, r.probs = verdicts[pid]

        # Inputs on which the non-overfitting candidates disagree = where the spec is silent.
        keep = [r.patch.patch_id for r in plausible if r.verdict != Label.OVERFITTING]
        if len(keep) >= 2:
            sub = cluster(matrix, keep)
            for j in sub.disagreement_inputs:
                groups: dict[str, list[str]] = {}
                for pid in keep:
                    groups.setdefault(matrix.candidates[pid][j], []).append(pid)
                result.ambiguity_witnesses.append((matrix.inputs[j], groups))

        result.n_inputs, result.clustering, result.matrix = len(inputs), clustering, matrix
        return result
