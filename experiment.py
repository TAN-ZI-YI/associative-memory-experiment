"""Run a fixed, synthetic retention study and record every intervention."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import torch
from torch.nn import functional as F

from memory import (
    DISTRACTORS, PAIRS, VOCABULARY, WIDTH, BindingStateOwner,
    DeltaBindingMemory, SymbolWorld, digest_model,
)


def score(answers, expected):
    correct = answers.eq(expected)
    return {
        "correct": int(correct.sum()), "total": correct.numel(),
        "fraction": float(correct.double().mean()),
        "all_queries_correct": int(correct.all(-1).sum()),
        "sequences": correct.shape[0],
    }


def dictionary_answers(events, cues, recent_only=False):
    """Transparent exact-match baseline; single symbols contain no pair."""
    answers = []
    for history, queries in zip(events.tolist(), cues.tolist()):
        table = {}
        for first, second in history[-DISTRACTORS:] if recent_only else history:
            if first and second:
                table[first] = second
                table[second] = first
        answers.append([table.get(query, 0) for query in queries])
    return torch.tensor(answers, dtype=torch.long)


@torch.no_grad()
def evaluate(model, seed, sequences=256):
    before = digest_model(model)
    case = SymbolWorld(seed + 20000).sample(sequences, heldout=True, counterfactual=True)
    state_a = model.write(case["events"])
    state_b = model.write(case["counterfactual_events"])
    zero = torch.zeros_like(state_a)
    cues = case["contrast_cues"]
    expected_a = case["contrast_targets"] + 1
    expected_b = case["counterfactual_targets"] + 1
    changed = expected_a.ne(expected_b)

    def read(state, query=cues):
        return model.read(state, query).argmax(-1) + 1

    answer_a, answer_b, answer_zero = read(state_a), read(state_b), read(zero)
    updated_state = model.write(torch.cat((case["correction"], case["events"][:, -DISTRACTORS:]), 1), state_a)
    answer_updated = read(updated_state)
    old_partner_cues = expected_a[:, :1]
    # After (a,b) is replaced by (a,new), b should no longer point to a.
    # This diagnostic measures retained stale links, rather than assuming
    # that successful forward replacement implements relational deletion.
    stale_reverse = read(updated_state, old_partner_cues).eq(cues[:, :1])
    metrics = {
        "heldout_both_directions": score(read(state_a, case["cues"]), case["targets"] + 1),
        "full_state_a": score(answer_a, expected_a),
        "full_state_b": score(answer_b, expected_b),
        "zero_state_vs_a": score(answer_zero, expected_a),
        "donor_state_vs_a": score(answer_b, expected_a),
        "donor_state_vs_donor": score(answer_b, expected_b),
        "changed_queries_follow_both_histories": {
            "correct": int((answer_a.eq(expected_a) & answer_b.eq(expected_b))[changed].sum()),
            "total": int(changed.sum()),
        },
        "new_forward_binding": score(answer_updated[:, :1], (case["corrected_targets"] + 1)[:, :1]),
        "unrelated_bindings_after_update": score(answer_updated[:, 1:], (case["corrected_targets"] + 1)[:, 1:]),
        "stale_reverse_link_retained": {"count": int(stale_reverse.sum()), "total": sequences},
        "exact_dictionary_full_history": score(dictionary_answers(case["events"], cues), expected_a),
        "exact_dictionary_recent_window": score(dictionary_answers(case["events"], cues, recent_only=True), expected_a),
    }
    assert torch.equal(case["events"][:, -DISTRACTORS:], case["counterfactual_events"][:, -DISTRACTORS:])
    assert torch.equal(case["events"][:, :PAIRS].flatten(1).sort(-1).values,
                       case["counterfactual_events"][:, :PAIRS].flatten(1).sort(-1).values)

    owner = BindingStateOwner("synthetic-subject", model)
    source = case["events"][0].clone()
    for index, event in enumerate(source):
        owner.observe(f"event-{index}", owner.subject_id, event)
    snapshot_before = owner.snapshot()["state"]
    expected_logits = model.read(snapshot_before, cues[:1])[0]
    source.zero_()
    owner_logits = owner.read(owner.subject_id, cues[0])
    assert torch.equal(expected_logits, owner_logits)
    assert torch.equal(snapshot_before, owner.snapshot()["state"])
    assert digest_model(model) == before

    # Keep the first four cases, with no filtering by correctness.
    examples = []
    for index in range(min(4, sequences)):
        examples.append({
            "seed": seed, "sequence": index,
            "early_a": case["events"][index, :PAIRS].tolist(),
            "early_b": case["counterfactual_events"][index, :PAIRS].tolist(),
            "recent_shared": case["events"][index, -DISTRACTORS:].tolist(),
            "cues": cues[index].tolist(),
            "expected_a": expected_a[index].tolist(),
            "expected_b": expected_b[index].tolist(),
            "answers_a": answer_a[index].tolist(),
            "answers_b": answer_b[index].tolist(),
            "answers_zero": answer_zero[index].tolist(),
            "update": case["correction"][index, 0].tolist(),
            "answers_updated": answer_updated[index].tolist(),
            "expected_updated": (case["corrected_targets"][index] + 1).tolist(),
            "old_partner_cue": int(old_partner_cues[index, 0]),
            "old_partner_answer_after_update": int(read(updated_state[index:index + 1], old_partner_cues[index:index + 1])[0, 0]),
            "state_a": state_a[index].tolist(), "state_b": state_b[index].tolist(),
            "state_updated": updated_state[index].tolist(),
        })
    return {
        "metrics": metrics,
        "controls": {"same_recent_inputs": True, "same_early_symbol_counts": True,
                     "same_model_parameters": True, "owner_source_free_read_max_difference": float((owner_logits - expected_logits).abs().max()),
                     "query_does_not_write_state": True},
        "examples": examples,
    }


def train(seed, steps, directory):
    torch.manual_seed(seed)
    model = DeltaBindingMemory()
    initial = digest_model(model)
    initial_observed = evaluate(model, seed)
    torch.save(model.state_dict(), directory / f"initial-{seed}.pt")
    world = SymbolWorld(seed + 1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    curve = []
    for step in range(steps):
        case = world.sample(64)
        optimizer.zero_grad(set_to_none=True)
        logits = model(case["events"], case["cues"])
        loss = F.cross_entropy(logits.flatten(0, 1), case["targets"].flatten())
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step == 0 or (step + 1) % 250 == 0:
            row = {"step": step + 1, "loss": float(loss.detach()),
                   "batch_accuracy": float(logits.argmax(-1).eq(case["targets"]).float().mean())}
            curve.append(row)
            print(json.dumps({"seed": seed, **row}), flush=True)
    model.eval()
    torch.save(model.state_dict(), directory / f"final-{seed}.pt")
    return model, {"initial_parameter_sha256": initial, "final_parameter_sha256": digest_model(model),
                   "pair_split_sha256": world.split_sha256, "training_curve": curve,
                   "untrained_metrics": initial_observed["metrics"]}


def run(output, steps=2000, seeds=(7, 19)):
    if not 1 <= steps <= 2000 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use 1..2000 steps and distinct seeds.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    root = Path(__file__).parent
    source_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                     for name in ["memory.py", "experiment.py", "test_memory.py"]}
    report = {
        "protocol": "symbolic-state-intervention-v1", "started_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "RUNNING", "source_sha256": source_hashes,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.system(), "machine": platform.machine(), "device": "cpu", "threads": 1},
        "config": {"seeds": list(seeds), "steps_per_seed": steps, "batch": 64,
                   "vocabulary": VOCABULARY, "width": WIDTH, "pairs": PAIRS, "distractors": DISTRACTORS,
                   "evaluation_sequences_per_seed": 256, "learning_rate": .002, "optimizer": "Adam",
                   "checkpoint_selection": "final_step_only", "pretrained_weights": False},
        "limitations": [
            "Evaluation holds out pair combinations within a known vocabulary.",
            "The task supplies pair boundaries; the model uses delta-rule updates.",
            "Single-symbol distractors are drawn from symbols outside the remembered pairs. Naturalistic interference remains a follow-up question.",
            "The complete-history dictionary also solves this task, providing an exact-storage baseline.",
            "Donor-state answers are scored against both histories to identify which contents determine recall.",
            "The stale-link diagnostic measures whether an old reverse association remains after a forward replacement.",
            "Results describe two fixed seeds. Statistical generalization requires additional runs.",
        ], "results": [],
    }
    # Freeze the protocol before training; evaluation never chooses a checkpoint.
    (output / "config.json").write_text(json.dumps(report, indent=2) + "\n")
    start = time.monotonic()
    for seed in seeds:
        model, provenance = train(seed, steps, output)
        observed = evaluate(model, seed)
        report["results"].append({"seed": seed, **provenance, **observed})
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"seed": seed, "metrics": observed["metrics"]}), flush=True)
    report.update(status="COMPLETE", elapsed_seconds=time.monotonic() - start,
                  finished_at=datetime.datetime.now().astimezone().isoformat())
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "COMPLETE", "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="A new directory; existing runs are never overwritten.")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 19])
    arguments = parser.parse_args()
    run(arguments.output, arguments.steps, tuple(arguments.seeds))
