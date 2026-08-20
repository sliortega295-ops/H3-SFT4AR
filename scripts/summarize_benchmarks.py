#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize H3 benchmark.json files.")
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.result_root.glob("*/benchmark.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "mode": path.parent.name,
                "world": data.get("world_size"),
                "steps": data.get("measured_steps_per_rank"),
                "max_s": data.get("elapsed_seconds_max_rank"),
                "step_s": data.get("steps_per_second"),
                "global_sample_s": data.get("samples_per_second_global"),
            }
        )
    if not rows:
        raise SystemExit(f"No */benchmark.json under {args.result_root}")

    baseline = next(
        (row for row in rows if row["mode"] in {"baseline", "zero-original"}),
        rows[0],
    )
    baseline_rate = baseline["global_sample_s"]
    print("| mode | world | steps/rank | max-rank seconds | global samples/s | speedup |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        rate = row["global_sample_s"]
        speedup = None if baseline_rate in (None, 0) or rate is None else rate / baseline_rate
        print(
            f"| {row['mode']} | {row['world']} | {row['steps']} | "
            f"{row['max_s']:.4f} | {rate:.6f} | "
            f"{('-' if speedup is None else f'{speedup:.3f}x')} |"
        )


if __name__ == "__main__":
    main()
