"""Run benchmark for a trained navigation model."""
from __future__ import annotations

import argparse

from drive2win.benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 99])
    args = parser.parse_args()
    result = run_benchmark(args.tag, f"nav_{args.tag}.npz", args.data, args.seeds)
    print("Benchmark summary:")
    for k, v in result["summary"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
