#!/usr/bin/env python3
"""Compare mAP JSON outputs from eval_map.py (e.g. edge vs server).

    python3 eval/compare_map.py benchmark/results/map_edge.json \
                                benchmark/results/map_server.json
"""
import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+", help="map_*.json files from eval_map.py")
    args = ap.parse_args()

    rows = [json.load(open(p)) for p in args.jsons]
    cols = ["label", "map50", "map50_95", "precision", "recall"]
    print("".join(f"{c:>12s}" for c in cols))
    print("-" * (12 * len(cols)))
    for r in rows:
        print("".join(f"{str(r.get(c, '')):>12s}" for c in cols))

    # accuracy delta (first = baseline)
    if len(rows) >= 2:
        base = rows[0]
        print()
        for r in rows[1:]:
            d = round(r.get("map50_95", 0) - base.get("map50_95", 0), 4)
            print(f"  {r['label']} vs {base['label']}: mAP@0.5:0.95 delta = {d:+}")


if __name__ == "__main__":
    main()
