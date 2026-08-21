#!/usr/bin/env python3
import argparse
import glob
import json
import math
from pathlib import Path


def load_records(run_dir, expected_world_size, expected_steps):
    paths = sorted(Path(path) for path in glob.glob(str(Path(run_dir) / "correctness.rank*.jsonl")))
    if len(paths) != expected_world_size:
        raise ValueError(
            f"{run_dir}: expected {expected_world_size} rank files, found {len(paths)}"
        )
    records = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rank_records = [json.loads(line) for line in handle if line.strip()]
        if len(rank_records) != expected_steps:
            raise ValueError(
                f"{path}: expected {expected_steps} records, found {len(rank_records)}"
            )
        for record in rank_records:
            key = (int(record["rank"]), int(record["step"]))
            if key in records:
                raise ValueError(f"{run_dir}: duplicate record {key}")
            records[key] = record
    expected_keys = {
        (rank, step)
        for rank in range(expected_world_size)
        for step in range(expected_steps)
    }
    if set(records) != expected_keys:
        raise ValueError(f"{run_dir}: rank/step grid is incomplete")
    return records


def close(left, right, *, rel_tol=1e-6, abs_tol=1e-6):
    return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)


def compare_signature(left, right, path, failures):
    if left is None or right is None:
        if left != right:
            failures.append(f"{path}: one signature is missing")
        return
    for field in ("dtype", "shape", "numel", "sampled_sha256"):
        if left.get(field) != right.get(field):
            failures.append(f"{path}.{field}: {left.get(field)!r} != {right.get(field)!r}")
    for field in ("sum", "sumsq", "absmax"):
        if field in left or field in right:
            if field not in left or field not in right or not close(left[field], right[field]):
                failures.append(f"{path}.{field}: {left.get(field)!r} != {right.get(field)!r}")


def compare_runs(reference, candidate, label):
    failures = []
    for key in sorted(reference):
        left = reference[key]
        right = candidate[key]
        prefix = f"{label}.rank{key[0]}.step{key[1]}"
        if left["sample"] != right["sample"]:
            failures.append(f"{prefix}.sample: {left['sample']} != {right['sample']}")
        if not close(left["loss"], right["loss"]):
            failures.append(f"{prefix}.loss: {left['loss']} != {right['loss']}")
        if len(left["predictions"]) != len(right["predictions"]):
            failures.append(f"{prefix}.predictions: output count differs")
        else:
            for index, (left_sig, right_sig) in enumerate(
                zip(left["predictions"], right["predictions"])
            ):
                compare_signature(
                    left_sig, right_sig, f"{prefix}.prediction{index}", failures
                )
        for field in ("parameter_before", "gradient", "parameter_after"):
            if set(left[field]) != set(right[field]):
                failures.append(f"{prefix}.{field}: selected parameter names differ")
                continue
            for name in sorted(left[field]):
                compare_signature(
                    left[field][name],
                    right[field][name],
                    f"{prefix}.{field}.{name}",
                    failures,
                )
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_a")
    parser.add_argument("baseline_b")
    parser.add_argument("optimized_zero")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline_a = load_records(args.baseline_a, args.world_size, args.steps)
    baseline_b = load_records(args.baseline_b, args.world_size, args.steps)
    optimized_zero = load_records(args.optimized_zero, args.world_size, args.steps)
    baseline_repeat_failures = compare_runs(
        baseline_a, baseline_b, "baseline_a_vs_b"
    )
    optimized_failures = compare_runs(
        baseline_a, optimized_zero, "baseline_a_vs_optimized_zero"
    )
    report = {
        "gate_pass": not baseline_repeat_failures and not optimized_failures,
        "contract": {
            "world_size": args.world_size,
            "optimizer_steps": args.steps,
            "fixed_sample_and_chunk_schedule": True,
            "comparison": "sample IDs, loss, prediction signatures, and three largest trainable local parameter/gradient shard signatures",
        },
        "baseline_repeat": {
            "pass": not baseline_repeat_failures,
            "failure_count": len(baseline_repeat_failures),
            "failures": baseline_repeat_failures[:100],
        },
        "optimized_zero": {
            "pass": not optimized_failures,
            "failure_count": len(optimized_failures),
            "failures": optimized_failures[:100],
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    raise SystemExit(0 if report["gate_pass"] else 1)


if __name__ == "__main__":
    main()
