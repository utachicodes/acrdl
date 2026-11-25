"""Training records, JSONL I/O, problem-level splits and batching."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from acrdl.types import Label

if TYPE_CHECKING:  # pragma: no cover
    from acrdl.pipeline import BugReport


@dataclass
class Record:
    """One labelled plausible patch with all model inputs."""

    problem_id: str
    bug_id: str
    patch_id: str
    code: str
    evidence: list[float]
    clusters: list[list[float]]
    own_cluster: int
    label: int


def problem_id_of(task_id: str) -> str:
    """Underspecified variants share the problem id of their source problem."""
    return task_id.split("~")[0]


def records_from_report(report: BugReport, labels: Mapping[str, Label]) -> list[Record]:
    from acrdl.evidence.code_text import code_text

    bug = report.bug
    out = []
    for r in report.patches:
        if not r.plausible or r.evidence is None or r.patch.patch_id not in labels:
            continue
        out.append(Record(
            problem_id=problem_id_of(bug.problem.task_id),
            bug_id=bug.bug_id,
            patch_id=r.patch.patch_id,
            code=code_text(bug.source, r.patch.source, bug.problem.prompt),
            evidence=list(r.evidence.vector),
            clusters=[list(row) for row in r.evidence.cluster_features],
            own_cluster=r.evidence.cluster_id,
            label=int(labels[r.patch.patch_id]),
        ))
    return out


def save_jsonl(records: Sequence[Record], path: Union[str, Path], append: bool = False) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a" if append else "w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")


def load_jsonl(path: Union[str, Path]) -> list[Record]:
    with Path(path).open() as f:
        return [Record(**json.loads(line)) for line in f if line.strip()]


def split_by_problem(records: Sequence[Record], fractions: tuple[float, float, float] = (
        0.7, 0.15, 0.15), seed: int = 13) -> tuple[list[Record], list[Record], list[Record]]:
    """70/15/15 train/val/test split with every problem in exactly one split.

    The split is stratified: problems are grouped by which rare labels they
    contain (ambiguous, overfitting, both or neither) and each group is split
    70/15/15 on its own, so the rare classes reach validation and test even
    when only a few problems have them.
    """
    labels_of: dict[str, set[int]] = defaultdict(set)
    for r in records:
        labels_of[r.problem_id].add(int(r.label))
    strata: dict[tuple[bool, bool], list[str]] = defaultdict(list)
    for pid in sorted(labels_of):
        ls = labels_of[pid]
        strata[(int(Label.AMBIGUOUS) in ls, int(Label.OVERFITTING) in ls)].append(pid)
    rng = random.Random(seed)
    train_p: set[str] = set()
    val_p: set[str] = set()
    for key in sorted(strata):
        group = strata[key]
        rng.shuffle(group)
        n = len(group)
        n_val = round(fractions[1] * n)
        n_test = round(fractions[2] * n)
        if n >= 3:  # every split gets at least one problem of each stratum
            n_val, n_test = max(1, n_val), max(1, n_test)
        n_train = n - n_val - n_test
        train_p.update(group[:n_train])
        val_p.update(group[n_train:n_train + n_val])
    train = [r for r in records if r.problem_id in train_p]
    val = [r for r in records if r.problem_id in val_p]
    test = [r for r in records if r.problem_id not in train_p | val_p]
    return train, val, test


def bug_batches(records: Sequence[Record], batch_size: int, seed: int = 0,
                shuffle: bool = True) -> Iterator[list[Record]]:
    """Batches that keep all candidates of a bug together (needed by SupCon)."""
    by_bug: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_bug[r.bug_id].append(r)
    bugs = list(by_bug)
    if shuffle:
        random.Random(seed).shuffle(bugs)
    batch: list[Record] = []
    for b in bugs:
        if batch and len(batch) + len(by_bug[b]) > batch_size:
            yield batch
            batch = []
        batch.extend(by_bug[b])
    if batch:
        yield batch


def collate(records: Sequence[Record]) -> dict[str, Any]:
    """Pad cluster sets and build tensors (requires torch)."""
    import torch

    k = max(len(r.clusters) for r in records)
    c = len(records[0].clusters[0])
    clusters = torch.zeros(len(records), k, c)
    mask = torch.ones(len(records), k, dtype=torch.bool)
    for i, r in enumerate(records):
        clusters[i, : len(r.clusters)] = torch.tensor(r.clusters)
        mask[i, : len(r.clusters)] = False
    bug_index = {b: i for i, b in enumerate(dict.fromkeys(r.bug_id for r in records))}
    return {
        "texts": [r.code for r in records],
        "evidence": torch.tensor([r.evidence for r in records], dtype=torch.float32),
        "clusters": clusters,
        "cluster_mask": mask,
        "own_index": torch.tensor([r.own_cluster for r in records], dtype=torch.long),
        "labels": torch.tensor([r.label for r in records], dtype=torch.long),
        "groups": torch.tensor([bug_index[r.bug_id] for r in records], dtype=torch.long),
    }
