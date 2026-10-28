"""Validate `/v1/decisions` answers, shapes and coexistence with generation via the OpenAI SDK.

Run against a separately started server. Each case has a known answer; the report records
per-type accuracy, the protocol shape of every answer, request latency, a batch of many questions
sharing one input, and a chat completion issued while decisions are in flight. The SDK
(`openai>=3.26`, which has `client.decisions`) is a validation dependency only.
"""

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEPARTMENTS = [
    {"value": "billing", "description": "Payments, invoices, charges and refunds."},
    {"value": "technical", "description": "Problems using the product or app."},
    {"value": "shipping", "description": "Delivery, tracking and lost parcels."},
    {"value": "other", "description": "Requests outside these categories."},
]
SEVERITY = [
    {"label": "Cosmetic", "description": "Appearance only; nothing stops working."},
    {"label": "Workaround available", "description": "A task fails, but another way works."},
    {"label": "Fully blocked", "description": "A task fails with no workaround."},
]
DAMAGE = "Does the customer report a damaged or broken item?"

# (input, question, expected) where expected is a bool, a choice value, or a level index.
CASES = [
    ("The package arrived with a broken screen.", ("predicate", DAMAGE), True),
    ("My order arrived on time and works perfectly, thanks!", ("predicate", DAMAGE), False),
    ("The mug was shattered when I opened the box.", ("predicate", DAMAGE), True),
    ("Can I change the delivery address for my order?", ("predicate", DAMAGE), False),
    ("The left hinge of the laptop snapped on arrival.", ("predicate", DAMAGE), True),
    ("I would like to know your store opening hours.", ("predicate", DAMAGE), False),
    ("I was charged twice for my order.", ("choice", DEPARTMENTS), "billing"),
    ("The app crashes every time I open settings.", ("choice", DEPARTMENTS), "technical"),
    ("My parcel has said 'in transit' for two weeks.", ("choice", DEPARTMENTS), "shipping"),
    ("Please refund the subscription fee from last month.", ("choice", DEPARTMENTS), "billing"),
    ("I cannot log in; the password reset link fails.", ("choice", DEPARTMENTS), "technical"),
    ("The courier left my package at the wrong house.", ("choice", DEPARTMENTS), "shipping"),
    ("Do you sponsor local football teams?", ("choice", DEPARTMENTS), "other"),
    ("The button colour is slightly off in dark mode.", ("score", SEVERITY), 0),
    ("Export fails in Safari but works in Chrome.", ("score", SEVERITY), 1),
    ("Nobody can sign in at all; the login service is down.", ("score", SEVERITY), 2),
    ("A tooltip has a typo in the word 'recieve'.", ("score", SEVERITY), 0),
    ("Upload fails from the web page, but the desktop app uploads fine.", ("score", SEVERITY), 1),
    ("Every payment attempt fails and there is no other way to pay.", ("score", SEVERITY), 2),
]


def question(kind, spec, name):
    if kind == "predicate":
        return {"type": "predicate", "name": name, "instructions": spec}
    if kind == "choice":
        return {
            "type": "choice",
            "name": name,
            "instructions": "Which department should handle this message?",
            "choices": spec,
        }
    return {
        "type": "score",
        "name": name,
        "instructions": "How severe is this issue?",
        "levels": spec,
    }


def correct(answer, expected):
    if answer.type == "predicate":
        return (answer.probability >= 0.5) == expected
    if answer.type == "choice":
        return answer.choice == expected
    return round(answer.score) == expected


def check_shape(answer, kind, spec):
    assert answer.type == kind, answer
    if kind == "predicate":
        assert 0.0 <= answer.probability <= 1.0
        return
    assert 0.0 <= answer.confidence <= 1.0
    total = sum(p.probability for p in answer.probabilities)
    assert abs(total - 1.0) < 1e-3, total
    if kind == "choice":
        assert [p.value for p in answer.probabilities] == [c["value"] for c in spec]
    else:
        assert [p.label for p in answer.probabilities] == [s["label"] for s in spec]
        assert 0.0 <= answer.score <= len(spec) - 1


def single_cases(client, model):
    results, latencies = [], []
    for index, (text, (kind, spec), expected) in enumerate(CASES):
        started = time.perf_counter()
        decision = client.decisions.create(
            model=model, input=text, questions=[question(kind, spec, f"q{index}")]
        )
        latencies.append(time.perf_counter() - started)
        answer = decision.answers[0]
        assert answer.name == f"q{index}", answer
        check_shape(answer, kind, spec)
        results.append(
            {
                "input": text,
                "type": kind,
                "expected": expected,
                "answer": answer.model_dump(),
                "correct": correct(answer, expected),
                "input_tokens": decision.usage.input_tokens,
            }
        )
    return results, latencies


def conversation_case(client, model):
    decision = client.decisions.create(
        model=model,
        input=[
            {"role": "developer", "content": "Judge only the final assistant turn."},
            {"role": "user", "content": "Please refund the duplicate charge."},
            {"type": "function_call", "call_id": "c1", "name": "lookup_order",
             "arguments": json.dumps({"order": 42})},
            {"type": "function_call_output", "call_id": "c1",
             "output": json.dumps({"charges": [49, 49]})},
            {"role": "user", "content": [
                {"type": "input_text", "text": "Order record:"},
                {"type": "input_json", "data": {"order": 42, "charges": [49, 49]}},
            ]},
            {"role": "assistant", "content": "I have refunded one of the two $49 charges."},
        ],
        questions=[
            {"type": "predicate", "name": "duplicate",
             "instructions": "Does the order record show a duplicate charge?"},
            {"type": "predicate", "name": "refused",
             "instructions": "Did the assistant refuse the customer's request?"},
        ],
    )
    answers = {answer.name: answer.probability for answer in decision.answers}
    return {
        "answers": answers,
        "correct": answers["duplicate"] >= 0.5 and answers["refused"] < 0.5,
    }


def batch_case(client, model, count):
    text = CASES[0][0]
    questions = [question("predicate", DAMAGE, f"b{i}") for i in range(count)]
    started = time.perf_counter()
    decision = client.decisions.create(model=model, input=text, questions=questions)
    elapsed = time.perf_counter() - started
    assert [a.name for a in decision.answers] == [q["name"] for q in questions]
    probabilities = [a.probability for a in decision.answers]
    return {
        "questions": count,
        "seconds": elapsed,
        "input_tokens": decision.usage.input_tokens,
        # Identical questions over identical input must agree.
        "probability_spread": max(probabilities) - min(probabilities),
    }


def coexistence_case(client, model):
    with ThreadPoolExecutor(max_workers=2) as pool:
        decision = pool.submit(batch_case, client, model, 32)
        chat = pool.submit(
            client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": "Name one primary colour."}],
            max_tokens=32,
        )
        chat = chat.result()
        return {
            "decisions": decision.result(),
            "chat_finish_reason": chat.choices[0].finish_reason,
            "chat_completion_tokens": chat.usage.completion_tokens,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default=None, help="deployment name; default /v1/models")
    parser.add_argument("--batch", type=int, default=64, help="questions in the shared-input batch")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="unused")
    model = args.model or client.models.list().data[0].id
    # The first read may capture device graphs; it is not part of the latency sample.
    client.decisions.create(
        model=model, input="warm up", questions=[question("predicate", DAMAGE, "warmup")]
    )
    results, latencies = single_cases(client, model)
    by_type = {}
    for result in results:
        total, right = by_type.get(result["type"], (0, 0))
        by_type[result["type"]] = (total + 1, right + int(result["correct"]))
    report = {
        "model": model,
        "accuracy": {kind: {"correct": right, "total": total}
                     for kind, (total, right) in by_type.items()},
        "single_question_latency_s": {
            "median": statistics.median(latencies),
            "max": max(latencies),
            "samples": len(latencies),
        },
        "conversation": conversation_case(client, model),
        "batch": batch_case(client, model, args.batch),
        "coexistence": coexistence_case(client, model),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
