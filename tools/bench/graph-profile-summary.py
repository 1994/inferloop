#!/usr/bin/env python3
"""Summarize per-node CUDA graph events by graph kind and operation.

Input is the JSONL a server writes when `INFER_CUDA_PREFILL_PROFILE` names a file. Event timing
perturbs throughput, so use it to rank where GPU time goes, never as a timing result.
"""

import argparse
import collections
import json
import sys
from pathlib import Path


def summarize(lines):
    """Per graph kind: replay count, total milliseconds and milliseconds per operation."""
    graphs = collections.defaultdict(
        lambda: {"replays": 0, "total_ms": 0.0, "ops": collections.Counter()}
    )
    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        graph = graphs[event.get("graph", "?")]
        graph["replays"] += 1
        graph["total_ms"] += event.get("total_ms", 0.0)
        graph["ops"].update(event.get("by_op") or {})
    return dict(graphs)


def render(graphs, top):
    total = sum(graph["total_ms"] for graph in graphs.values())
    lines = [f"all graphs {total / 1e3:.3f} s"]
    for name, graph in sorted(graphs.items(), key=lambda item: -item[1]["total_ms"]):
        share = graph["total_ms"] / total * 100 if total else 0.0
        seconds = graph["total_ms"] / 1e3
        lines.append(f"{name}: {seconds:.3f} s over {graph['replays']} replays ({share:.1f}%)")
        for op, ms in graph["ops"].most_common(top):
            lines.append(
                f"  {op:12s} {ms / 1e3:7.3f} s  {ms / graph['total_ms'] * 100:5.1f}%"
                f"  {ms / graph['replays']:.3f} ms/replay"
            )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path, help="INFER_CUDA_PREFILL_PROFILE JSONL")
    parser.add_argument("--top", type=int, default=10, help="operations listed per graph")
    args = parser.parse_args()
    with args.events.open() as events:
        print(render(summarize(events), args.top))


if __name__ == "__main__":
    sys.exit(main())
