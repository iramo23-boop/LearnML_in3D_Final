"""Compare all benchmark iterations."""
from __future__ import annotations

import json
from pathlib import Path

from drive2win.viz import plot_iteration_history


def main() -> None:
    rows = []
    for p in sorted(Path("benchmarks").glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        s = obj["summary"]
        rows.append([obj["tag"], s["complete_runs"], s["total_runs"], s["mean_checkpoints"], s["mean_crashes"]])

    if not rows:
        print("No benchmark JSON files found.")
        return

    print("tag | complete | runs | mean_checkpoints | mean_crashes")
    print("-" * 62)
    for r in rows:
        print(f"{r[0]:<12} {r[1]:>3}/{r[2]:<3} {r[3]:>8.2f} {r[4]:>14.2f}")
    plot_iteration_history("benchmarks", "benchmarks/iteration_history.png")
    print("Saved benchmarks/iteration_history.png")


if __name__ == "__main__":
    main()
