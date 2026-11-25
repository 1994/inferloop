#!/usr/bin/env python3
"""Replay recorded multi-round agent traces against a running server and break down the time.

A trace is a JSON file holding `instructions`, function `tools` and `rounds`. Each round lists
the items that arrive before one model call (the first user message, or the tool outputs of
the previous call) and `tool_latency_ms`, the time a mocked tool takes before those items exist.

Every round is one streamed `/v1/responses` call at temperature 0. By default the model's own
output becomes history (`--history generated`), which is what an agent harness does and what
lets engine-side reuse show up; recorded tool outputs are rebound to the model's call ids in
call order. A trace stops as `diverged` when the model makes fewer calls than recorded.
`--history recorded` appends the recorded outputs instead, so every build sees identical input
and a truncated response does not end the trace.
`--chain` sends `previous_response_id` and only the new items (server needs
`--enable-response-store`); otherwise the full input is resent each round.

Per round it reports queue+prefill time (to the first streamed delta), decode time, mocked tool
wait and the remainder, token usage, prefix tokens reused (from native observability, so run
one trace at a time against an otherwise idle server), and how many prefill tokens were already
part of the previous round's context ("repeated"). The copyable share counts output characters
covered by runs of at least `--copy-span` characters already present in the context, a
token-free estimate of what copy-from-context speculation could draft.

Run inside `tools/bench/safe-run.sh` against a release server; results are written as JSON.
"""

import argparse
import http.client
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_COPY_SPAN = 32
REQUEST_TIMEOUT_SECONDS = 600
OBSERVABILITY_ROUTE = "/native/v1/observability"
FIRST_TOKEN_EVENTS = (
    "response.output_text.delta",
    "response.reasoning_text.delta",
    "response.function_call_arguments.delta",
)
TERMINAL_EVENTS = ("response.completed", "response.incomplete", "response.failed")
MILLISECONDS_PER_SECOND = 1000


def connection(base):
    parts = urlsplit(base)
    return http.client.HTTPConnection(parts.hostname, parts.port, timeout=REQUEST_TIMEOUT_SECONDS)


def get_json(base, route):
    client = connection(base)
    try:
        client.request("GET", route)
        response = client.getresponse()
        body = response.read()
        if response.status != http.HTTPStatus.OK:
            raise RuntimeError(f"GET {route} returned {response.status}: {body[:200]!r}")
        return json.loads(body)
    finally:
        client.close()


def prefix_tokens_reused(base):
    """The engine's cumulative reused prefix tokens, or None when the route is unavailable."""
    try:
        return get_json(base, OBSERVABILITY_ROUTE).get("metrics", {}).get("prefix_tokens_reused")
    except (OSError, RuntimeError, ValueError):
        return None


def sse_events(lines):
    """Yield `(event, data)` pairs from an iterator of SSE byte lines."""
    event, data = None, []
    for raw in lines:
        line = raw.decode().rstrip("\r\n")
        if not line:
            if data:
                yield event, json.loads("\n".join(data))
            event, data = None, []
        elif line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:") :].strip())
    if data:
        yield event, json.loads("\n".join(data))


def stream_response(base, payload, clock=time.perf_counter):
    """Send one streamed Responses call; return the final response and its timings in seconds."""
    client = connection(base)
    started = clock()
    first = None
    try:
        client.request(
            "POST",
            "/v1/responses",
            body=json.dumps({**payload, "stream": True}),
            headers={"content-type": "application/json", "accept": "text/event-stream"},
        )
        response = client.getresponse()
        if response.status != http.HTTPStatus.OK:
            raise RuntimeError(
                f"/v1/responses returned {response.status}: {response.read()[:500]!r}"
            )
        for event, data in sse_events(response):
            kind = event or data.get("type")
            if first is None and kind in FIRST_TOKEN_EVENTS:
                first = clock()
            if kind in TERMINAL_EVENTS:
                finished = clock()
                first = finished if first is None else first
                return data["response"], {"first": first - started, "total": finished - started}
    finally:
        client.close()
    raise RuntimeError("stream ended without a terminal event")


def function_calls(items):
    return [item for item in items if item.get("type") == "function_call"]


def rebind_outputs(recorded, calls):
    """Recorded tool outputs with call ids replaced by the model's ids, in call order.

    Returns None when the model made fewer calls than the recorded outputs need.
    """
    outputs = [item for item in recorded if item.get("type") == "function_call_output"]
    if len(calls) < len(outputs):
        return None
    rebound, index = [], 0
    for item in recorded:
        if item.get("type") == "function_call_output":
            rebound.append({**item, "call_id": calls[index]["call_id"]})
            index += 1
        else:
            rebound.append(item)
    return rebound


def history_items(output):
    """Response output items that a harness feeds back, without server-only fields."""
    kept = []
    for item in output:
        if item.get("type") == "function_call":
            kept.append(
                {key: item[key] for key in ("type", "call_id", "name", "arguments") if key in item}
            )
        elif item.get("type") == "message":
            kept.append(
                {
                    "type": "message",
                    "role": item.get("role", "assistant"),
                    "content": item["content"],
                }
            )
        elif item.get("type") == "reasoning":
            kept.append(item)
    return kept


def item_text(item):
    if item.get("type") == "function_call":
        return item.get("arguments", "")
    if item.get("type") == "function_call_output":
        output = item.get("output", "")
        return (
            output if isinstance(output, str) else "".join(part.get("text", "") for part in output)
        )
    content = item.get("content", "")
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def visible_output_text(output):
    """Text the model wrote outside reasoning: messages and tool arguments."""
    return "".join(
        item_text(item) for item in output if item.get("type") in ("message", "function_call")
    )


def copyable_characters(output, context, span):
    """Characters of `output` covered by runs of at least `span` characters found in `context`.

    Greedy left-to-right extension, matching how a copy drafter would continue a hit.
    """
    if span <= 0 or len(output) < span:
        return 0
    covered, position = 0, 0
    while position + span <= len(output):
        if output[position : position + span] not in context:
            position += 1
            continue
        end = position + span
        while end < len(output) and output[position : end + 1] in context:
            end += 1
        covered += end - position
        position = end
    return covered


def round_record(index, usage, timings, tool_wait, reused, previous_context_tokens, copy):
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    reasoning = usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
    recomputed = None if reused is None else max(input_tokens - reused, 0)
    new_tokens = max(input_tokens - previous_context_tokens, 0)
    repeated = None if recomputed is None else max(recomputed - new_tokens, 0)
    first, total = timings["first"], timings["total"]
    return {
        "round": index,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning,
        "new_input_tokens": new_tokens,
        "prefix_tokens_reused": reused,
        "prefill_tokens_recomputed": recomputed,
        "prefill_tokens_repeated": repeated,
        "first_delta_ms": first * MILLISECONDS_PER_SECOND,
        "decode_ms": (total - first) * MILLISECONDS_PER_SECOND,
        "tool_wait_ms": tool_wait * MILLISECONDS_PER_SECOND,
        "copyable_characters": copy[0],
        "visible_output_characters": copy[1],
    }


def replay(base, model, trace, args, sleep=None):
    """Run one trace; return its rounds and how it ended. `sleep` mocks the tool latency."""
    sleep = sleep or time.sleep
    history, rounds, previous_id = [], [], None
    previous_context_tokens, pending = 0, None
    started = time.perf_counter()
    status = "completed"
    for index, recorded in enumerate(trace["rounds"]):
        new_items = recorded["input"]
        if pending is not None:
            new_items = (
                rebind_outputs(new_items, pending) if args.history == "generated" else new_items
            )
            if new_items is None:
                status = "diverged"
                break
        tool_wait = recorded.get("tool_latency_ms", 0) / MILLISECONDS_PER_SECOND
        sleep(tool_wait)
        history.extend(new_items)
        payload = {
            "model": model,
            "instructions": trace.get("instructions"),
            "tools": trace.get("tools", []),
            "temperature": 0,
            "max_output_tokens": args.max_output_tokens,
            "store": args.chain,
            "input": new_items if args.chain and previous_id else history,
        }
        if args.chain and previous_id:
            payload["previous_response_id"] = previous_id
        before = prefix_tokens_reused(base)
        response, timings = stream_response(base, payload)
        after = prefix_tokens_reused(base)
        reused = None if before is None or after is None else after - before
        output = response.get("output", [])
        produced = (
            history_items(output) if args.history == "generated" else recorded.get("output", [])
        )
        context = "".join(item_text(item) for item in history)
        text = visible_output_text(output)
        copy = (copyable_characters(text, context, args.copy_span), len(text))
        usage = response.get("usage") or {}
        entry = round_record(
            index, usage, timings, tool_wait, reused, previous_context_tokens, copy
        )
        entry["response_id"] = response.get("id")
        entry["status"] = response.get("status")
        entry["output_items"] = [item.get("name", item.get("type")) for item in output]
        rounds.append(entry)
        # Recorded outputs have no token count, so recorded replays credit only the prior input.
        previous_context_tokens = usage.get("input_tokens", 0)
        if args.history == "generated":
            previous_context_tokens += usage.get("output_tokens", 0)
        history.extend(produced)
        previous_id = response.get("id")
        pending = function_calls(produced)
        outcome = response.get("status", "failed")
        if outcome == "failed" or (outcome != "completed" and args.history == "generated"):
            status = outcome
            break
    wall = (time.perf_counter() - started) * MILLISECONDS_PER_SECOND
    return {"name": trace.get("name"), "status": status, "wall_ms": wall, "rounds": rounds}


def summarize(results):
    rounds = [entry for result in results for entry in result["rounds"]]
    wall = sum(result["wall_ms"] for result in results)

    def total(field):
        values = [entry[field] for entry in rounds if entry[field] is not None]
        return sum(values) if len(values) == len(rounds) else None

    first, decode, tool = total("first_delta_ms"), total("decode_ms"), total("tool_wait_ms")
    output_tokens, input_tokens = total("output_tokens"), total("input_tokens")
    visible = total("visible_output_characters")

    def share(part, whole):
        return None if part is None or not whole else part / whole

    return {
        "traces": len(results),
        "rounds": len(rounds),
        "statuses": sorted({result["status"] for result in results}),
        "wall_ms": wall,
        "time_share": {
            "queue_and_prefill": share(first, wall),
            "decode": share(decode, wall),
            "tool_wait": share(tool, wall),
            "other": share(wall - first - decode - tool, wall),
        },
        "input_tokens": input_tokens,
        "prefix_tokens_reused": total("prefix_tokens_reused"),
        "prefill_tokens_repeated": total("prefill_tokens_repeated"),
        "repeated_share_of_input": share(total("prefill_tokens_repeated"), input_tokens),
        "output_tokens": output_tokens,
        "reasoning_share_of_output": share(total("reasoning_tokens"), output_tokens),
        "copyable_share_of_visible_output": share(total("copyable_characters"), visible),
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="server base URL")
    parser.add_argument("--model", help="model id; defaults to the server's first model")
    parser.add_argument(
        "--trace", type=Path, action="append", required=True, help="trace JSON file"
    )
    parser.add_argument("--history", choices=("generated", "recorded"), default="generated")
    parser.add_argument("--chain", action="store_true", help="use previous_response_id chaining")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--copy-span", type=int, default=DEFAULT_COPY_SPAN)
    parser.add_argument("--repeats", type=int, default=1, help="replays per trace")
    parser.add_argument("--output", type=Path, required=True, help="result JSON path")
    args = parser.parse_args(argv)
    if args.chain and args.history == "recorded":
        # The server chains its own outputs, so recorded tool outputs would answer unknown calls.
        parser.error("--chain needs --history generated")
    return args


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    model = args.model or get_json(args.base, "/v1/models")["data"][0]["id"]
    traces = [json.loads(path.read_text()) for path in args.trace]
    results = [
        replay(args.base, model, trace, args) for _ in range(args.repeats) for trace in traces
    ]
    report = {
        "base": args.base,
        "model": model,
        "history": args.history,
        "chain": args.chain,
        "max_output_tokens": args.max_output_tokens,
        "summary": summarize(results),
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
