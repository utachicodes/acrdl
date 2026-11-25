import json
from pathlib import Path

import numpy as np

from acrdl.baselines import disagreement, patch_sim, tests_only
from acrdl.cli import main, toy_bugs
from acrdl.config import load_config
from acrdl.data import toy
from acrdl.generation.generator import MockGenerator
from acrdl.generation.prompts import extract_code, repair_prompt
from acrdl.pipeline import Pipeline
from acrdl.training.dataset import load_jsonl, split_by_problem
from acrdl.types import Label


def test_average_flagged_ambiguous(cfg, average_bug):
    rep = Pipeline(cfg, MockGenerator(toy.SCRIPTED_CANDIDATES)).run(average_bug)
    verdicts = [r.verdict for r in rep.patches]
    assert verdicts == [Label.AMBIGUOUS] * 4 + [Label.OVERFITTING]
    inputs = [args for args, _ in rep.ambiguity_witnesses]
    assert ([],) in inputs
    groups = next(g for args, g in rep.ambiguity_witnesses if args == ([],))
    assert set(groups) == {"ok:0.0", "error:ZeroDivisionError", "ok:None"}


def test_full_spec_average_is_correct(cfg):
    bug = next(b for b in toy_bugs(False) if b.problem.task_id == "toy/average")
    rep = Pipeline(cfg, MockGenerator(toy.SCRIPTED_CANDIDATES)).run(bug)
    plausible = [r for r in rep.patches if r.plausible]
    assert [r.verdict for r in plausible] == [Label.CORRECT, Label.CORRECT, Label.OVERFITTING]
    assert not rep.ambiguity_witnesses


def test_prompts():
    bug = toy_bugs(False)[0]
    assert "Buggy implementation" in repair_prompt(bug, ["average([1, 2]) == 1.5"])
    assert extract_code("text\n```python\ndef f():\n    pass\n```\n") == "def f():\n    pass\n"


def test_config_roundtrip():
    cfg = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    assert cfg.model.evidence_dim == 30 and cfg.decision.target_false_accept_rate == 0.05


def test_cli_demo_and_dataset(tmp_path, capsys):
    assert main(["demo", "--task", "toy/average", "--fuzz", "40"]) == 0
    out = capsys.readouterr().out
    assert "spec ambiguous at input ([],)" in out and "AMBIGUOUS" in out

    path = tmp_path / "toy.jsonl"
    assert main(["build-dataset", "--source", "toy", "--out", str(path)]) == 0
    records = load_jsonl(path)
    labels = {r.label for r in records}
    assert labels == {int(Label.CORRECT), int(Label.OVERFITTING), int(Label.AMBIGUOUS)}
    assert all(len(r.evidence) == 30 for r in records)

    train, val, test = split_by_problem(records, seed=0)
    assert not ({r.problem_id for r in train} & {r.problem_id for r in test})
    assert len(train) + len(val) + len(test) == len(records)

    y = np.array([r.label for r in records])
    assert (tests_only.predict_records(records) == Label.CORRECT).all()
    assert len(patch_sim.predict_records(records)) == len(records)
    pred = disagreement.predict_records(records)
    assert (pred == y).mean() > 0.6

    capsys.readouterr()
    assert main(["evaluate", "--data", str(path), "--baseline", "disagreement",
                 "--split", "all"]) == 0
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["macro_f1"] > 0.5
