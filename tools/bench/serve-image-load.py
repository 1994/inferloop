#!/usr/bin/env python3
"""Measure one closed-loop image-chat workload against an existing HTTP server.

Use unique, pre-generated images and the same manifest for each engine.  This is an
exploratory client, not the text serving gate or proof of equivalent model quality.
Start the server under the repository's protected benchmark workflow and use
``hardware-monitor.py`` for server CPU, RSS and GPU telemetry.  This client never
starts a server or initializes a GPU.  A completed report includes every rejection;
a nonzero exit means at least one warmup or measured request failed the contract.

Example:
  python3 tools/bench/serve-image-load.py --url http://127.0.0.1:8000 \\
    --model served-name --engine-label native --fixtures artifacts/image-fixtures \\
    --size 256x256 \\
    --concurrency 8 --samples 32 --warmup 4 --max-tokens 16 \\
    --require-output-tokens 16 --stream --output artifacts/image-run.json

A fixed output length is *checked*, not forced: servers that stop at EOS early are
reported as length mismatches, not silently compared as an equivalent workload.
An optional reference report checks observable request and output consistency;
it does not verify model weights, server configuration or numerical equivalence.
For a second engine, pass ``--reference-report`` with the first report's path;
inconsistent or missing evidence causes a nonzero exit after saving the report.

Connections use numeric addresses only (``localhost`` maps to 127.0.0.1). For a
hostname use ``--connect-ip``; HTTP Host and TLS verification retain the URL host.
Redirects are reported as HTTP errors. The request deadline includes connection,
TLS, sending and receiving; scheduling and bounded parsing may delay enforcement.
Streaming success requires a first visible text delta and a completed response.
"""

import argparse
import asyncio
import base64
import collections
import hashlib
import importlib.util
import io
import ipaddress
import json
import math
import mimetypes
import socket
import ssl
import statistics
import time
import urllib.parse
from pathlib import Path

MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_FIXTURE_BYTES = 24 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MIB_BYTES = 1024 * 1024
DEFAULT_MAX_PREPARED_MIB = 512
MAX_REQUEST_METADATA_BYTES = 64 * 1024
MAX_HEADER_BYTES = 64 * 1024
MAX_LINE_BYTES = 8 * 1024
READ_CHUNK_BYTES = 16 * 1024
SEND_CHUNK_BYTES = 64 * 1024
TRANSPORT_CLOSE_SECONDS = 0.1
MAX_CLIENT_CONCURRENCY = 256
TEXT_FINISH_REASONS = frozenset(("stop", "length"))
TEXT_CAPTURE_CONTRACT = "image-chat-text-v5"


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def fixture_payloads(args):
    manifest_path = args.fixtures / "manifest.json"
    with manifest_path.open("rb") as file:
        manifest_bytes = file.read(MAX_MANIFEST_BYTES + 1)
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ValueError("fixture manifest exceeds client limit")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    try:
        available = manifest["items"][args.size]
    except (KeyError, TypeError) as error:
        raise ValueError(f"fixture size not present: {args.size}") from error
    total = (args.warmup + args.samples) * args.images_per_request
    entries = available[args.offset : args.offset + total]
    if len(entries) != total:
        raise ValueError(f"need {total} fixture images from offset {args.offset}")
    root = args.fixtures.resolve()
    fixed = len(request_body(args, []))
    if fixed > MAX_REQUEST_METADATA_BYTES:
        raise ValueError("request metadata exceeds client limit")
    records = []
    sources = 4 * len(manifest_bytes)
    body_total = current_body = maximum_body = maximum_temporary = 0
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise ValueError("invalid fixture entry")
        path = (root / entry["file"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("fixture must be a file inside the fixture directory")
        size = path.stat().st_size
        if size > MAX_FIXTURE_BYTES:
            raise ValueError("fixture exceeds per-file client limit")
        if args.image_source == "url":
            source = entry.get("url")
            if not isinstance(source, str) or not source.startswith(("http://", "https://")):
                raise ValueError("URL mode requires an HTTP(S) url in every fixture entry")
            source_chars = len(source)
            source_body_bound = 6 * source_chars
            temporary = size
        else:
            mime = mimetypes.guess_type(path.name)[0]
            if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
                raise ValueError(f"unsupported image fixture type: {path.name}")
            encoded = ((size + 2) // 3) * 4
            source_chars = len(f"data:{mime};base64,") + encoded
            source_body_bound = source_chars
            temporary = size + 5 * encoded
        sources += 4 * (source_chars + len(entry["file"]) + 64) + 512
        maximum_temporary = max(maximum_temporary, temporary)
        current_body += source_body_bound + 128
        if (index + 1) % args.images_per_request == 0:
            body_bound = fixed + current_body
            body_total += body_bound
            maximum_body = max(maximum_body, body_bound)
            current_body = 0
        if (
            max(sources + maximum_temporary, sources + body_total + 4 * maximum_body)
            > args.max_prepared_mib * MIB_BYTES
        ):
            raise ValueError("fixture workload exceeds client preparation limit")
        records.append((entry, path, size))
    prepared, seen = [], set()
    for entry, path, size in records:
        with path.open("rb") as file:
            raw = file.read(size + 1)
        if len(raw) != size:
            raise ValueError("fixture size changed during preparation")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"fixture hash mismatch: {entry['file']}")
        if digest in seen:
            raise ValueError("fixture images must be unique across warmup and samples")
        seen.add(digest)
        if args.image_source == "url":
            source = entry.get("url")
            if not isinstance(source, str) or not source.startswith(("http://", "https://")):
                raise ValueError("URL mode requires an HTTP(S) url in every fixture entry")
        else:
            mime = mimetypes.guess_type(path.name)[0]
            if mime not in ("image/png", "image/jpeg", "image/webp", "image/gif"):
                raise ValueError(f"unsupported image fixture type: {path.name}")
            source = f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
        prepared.append((entry["file"], digest, source))
    return manifest, manifest_sha256, prepared


def request_body(args, group):
    content = [{"type": "image_url", "image_url": {"url": source}} for _, _, source in group]
    content.append({"type": "text", "text": args.prompt})
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": args.stream,
    }
    if args.stream:
        payload["stream_options"] = {"include_usage": True}
    if args.processor_options is not None:
        payload["mm_processor_kwargs"] = args.processor_options
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()


def read_limited(response):
    data = response.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError("response exceeds client capture limit")
    return data


def require_text_fields(value, allowed):
    if not isinstance(value, dict):
        raise ValueError("unsupported non-object response message")
    if value.get("role") not in (None, "assistant"):
        raise ValueError("unsupported response role")
    for key, item in value.items():
        if key not in allowed and item is not None and item != "" and item != []:
            raise ValueError(f"unsupported non-text response field: {key}")


def response_identity(item):
    response_id = item.get("id")
    model = item.get("model")
    if not isinstance(response_id, str) or not response_id:
        raise ValueError("unsupported response without an identity")
    if not isinstance(model, str) or not model:
        raise ValueError("unsupported response without a model")
    return response_id, model


class StreamCapture:
    """Capture text events incrementally while bounding both total and incomplete lines."""

    def __init__(self, started):
        self.started = started
        self.chunks = []
        self.usage = self.finish_reason = self.first = self.last = self.identity = None
        self.visible_events = 0
        self.read_time = None
        self.done = False
        self.total_bytes = 0
        self.pending = bytearray()
        self.frame_open = False
        self.data_lines = []

    def feed(self, data):
        # Events delivered together have the same observable arrival time.
        self.read_time = None
        self.total_bytes += len(data)
        if self.total_bytes > MAX_RESPONSE_BYTES:
            raise ValueError("stream exceeds client capture limit")
        scan_from = len(self.pending)
        self.pending.extend(data)
        while (end := self.pending.find(b"\n", scan_from)) >= 0:
            line = bytes(self.pending[:end])
            del self.pending[: end + 1]
            self.line(line.removesuffix(b"\r"))
            scan_from = 0

    def line(self, line):
        if not line:
            self.frame_open = False
            if self.data_lines:
                data = b"\n".join(self.data_lines)
                self.data_lines.clear()
                self.event(data)
            return
        self.frame_open = True
        if line.startswith(b":"):
            return
        field, separator, value = line.partition(b":")
        if field == b"data":
            self.data_lines.append(value.removeprefix(b" ") if separator else b"")

    def event(self, data):
        if self.done:
            raise ValueError("stream contains data after [DONE]")
        if data.strip() == b"[DONE]":
            self.done = True
            return
        item = json.loads(data)
        if item.get("error"):
            raise ValueError(f"stream error: {item['error']}")
        current_identity = response_identity(item)
        if self.identity is not None and self.identity != current_identity:
            raise ValueError("stream response identity changed")
        self.identity = current_identity
        self.usage = item.get("usage") or self.usage
        choices = item.get("choices") or []
        if choices:
            if len(choices) != 1:
                raise ValueError("unsupported multiple stream choices")
            choice = choices[0]
            if (
                type(choice.get("index")) is not int
                or choice["index"] != 0
                or self.finish_reason is not None
            ):
                raise ValueError("unsupported stream choice or additional output after finish")
            delta = choice.get("delta")
            require_text_fields(delta, {"role", "content"})
            text = delta.get("content")
            if text is None:
                text = ""
            if not isinstance(text, str):
                raise ValueError("non-string stream content")
            if text:
                if self.read_time is None:
                    self.read_time = (time.perf_counter() - self.started) * 1000
                self.last = self.read_time
                if self.first is None:
                    self.first = self.last
                self.visible_events += 1
                self.chunks.append(text)
            self.finish_reason = choice.get("finish_reason") or self.finish_reason

    def result(self):
        if self.pending or self.frame_open:
            raise ValueError("incomplete stream: unterminated SSE event")
        if not self.done or self.finish_reason is None:
            raise ValueError("incomplete stream: missing finish reason or [DONE]")
        if not isinstance(self.finish_reason, str) or self.finish_reason not in TEXT_FINISH_REASONS:
            raise ValueError(f"unsupported stream finish reason: {self.finish_reason}")
        if self.first is None or not valid_latency(self.first):
            raise ValueError("stream has no valid first visible content time")
        tokens = self.usage.get("completion_tokens") if isinstance(self.usage, dict) else None
        spacing = None
        if (
            self.visible_events > 1
            and self.last > self.first
            and type(tokens) is int
            and tokens > 1
        ):
            spacing = (self.last - self.first) / (tokens - 1)
        return {
            "text": "".join(self.chunks),
            "finish_reason": self.finish_reason,
            "response_model": self.identity[1],
            "usage": self.usage,
            "ttft_ms": self.first,
            "last_visible_content_ms": self.last,
            "visible_content_events": self.visible_events,
            "tpot_estimate_ms": spacing,
        }


def parse_stream(response, started):
    capture = StreamCapture(started)
    while True:
        data = response.read(min(READ_CHUNK_BYTES, MAX_RESPONSE_BYTES - capture.total_bytes + 1))
        if not data:
            break
        capture.feed(data)
    return capture.result()


def parse_complete(response):
    item = json.loads(read_limited(response))
    _, response_model = response_identity(item)
    choices = item.get("choices") or []
    if (
        len(choices) != 1
        or type(choices[0].get("index")) is not int
        or choices[0]["index"] != 0
        or not choices[0].get("finish_reason")
    ):
        raise ValueError("response lacks one completed choice")
    if (
        not isinstance(choices[0]["finish_reason"], str)
        or choices[0]["finish_reason"] not in TEXT_FINISH_REASONS
    ):
        raise ValueError(f"unsupported response finish reason: {choices[0]['finish_reason']}")
    message = choices[0].get("message")
    require_text_fields(message, {"role", "content"})
    if "content" not in message:
        raise ValueError("response message lacks content")
    text = message.get("content")
    if text is not None and not isinstance(text, str):
        raise ValueError("non-string response content")
    return {
        "text": text or "",
        "finish_reason": choices[0]["finish_reason"],
        "response_model": response_model,
        "usage": item.get("usage"),
        "ttft_ms": None,
        "last_visible_content_ms": None,
        "visible_content_events": 0,
        "tpot_estimate_ms": None,
    }


def server_address(host, connect_ip=None):
    """Use a numeric address so no unbounded system DNS call can delay a request."""
    value = connect_ip or ("127.0.0.1" if host.lower() == "localhost" else host)
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise ValueError("server hostname requires an explicit numeric --connect-ip") from error
    if getattr(address, "scope_id", None) is not None:
        raise ValueError("scoped IPv6 addresses are not supported")
    family = socket.AF_INET if address.version == 4 else socket.AF_INET6
    return str(address), family


def resolve_server(url, connect_ip=None):
    """Bind a numeric target while retaining the URL hostname for HTTP Host and TLS SNI."""
    target = urllib.parse.urlsplit(url)
    port = target.port or (443 if target.scheme == "https" else 80)
    address, family = server_address(target.hostname, connect_ip)
    addresses = [(family, socket.SOCK_STREAM, 0, "", (address, port))]
    return target, addresses, ssl.create_default_context() if target.scheme == "https" else None


async def read_line(reader):
    try:
        line = await reader.readline()
    except ValueError as error:
        raise ValueError("HTTP header or chunk line exceeds limit") from error
    if not line or not line.endswith(b"\n"):
        raise ValueError("incomplete HTTP header or chunk line")
    return line


async def read_headers(reader):
    size = 0
    while True:
        status = await read_line(reader)
        size += len(status)
        parts = status.rstrip(b"\r\n").split(b" ", 2)
        if len(parts) < 2 or not parts[0].startswith(b"HTTP/") or not parts[1].isdigit():
            raise ValueError("invalid HTTP status line")
        code = int(parts[1])
        headers = {}
        while True:
            line = await read_line(reader)
            size += len(line)
            if size > MAX_HEADER_BYTES:
                raise ValueError("HTTP headers exceed client capture limit")
            if line in (b"\r\n", b"\n"):
                break
            key, separator, value = line.partition(b":")
            if not separator:
                raise ValueError("invalid HTTP header")
            key = key.strip().lower()
            if key in headers and key in (b"content-length", b"transfer-encoding"):
                raise ValueError("ambiguous HTTP response framing")
            headers[key] = value.strip().lower()
        if not 100 <= code < 200:
            return code, headers


async def body_parts(reader, headers):
    """Yield bounded body chunks without letting a newline or chunk grow unchecked."""
    transfer = headers.get(b"transfer-encoding")
    length = headers.get(b"content-length")
    if transfer and length:
        raise ValueError("ambiguous HTTP response framing")
    if transfer:
        if transfer != b"chunked":
            raise ValueError("unsupported HTTP transfer encoding")
        while True:
            line = await read_line(reader)
            if not line.endswith(b"\r\n"):
                raise ValueError("invalid HTTP chunk line ending")
            digits = line[:-2].split(b";", 1)[0]
            if not digits or any(byte not in b"0123456789abcdefABCDEF" for byte in digits):
                raise ValueError("invalid HTTP chunk size")
            size = int(digits, 16)
            if size > MAX_RESPONSE_BYTES:
                raise ValueError("response exceeds client capture limit")
            if size == 0:
                trailer_size = 0
                while True:
                    trailer = await read_line(reader)
                    trailer_size += len(trailer)
                    if trailer_size > MAX_HEADER_BYTES:
                        raise ValueError("HTTP trailers exceed client capture limit")
                    if not trailer.endswith(b"\r\n"):
                        raise ValueError("invalid HTTP trailer line ending")
                    if trailer == b"\r\n":
                        return
            while size:
                part = await reader.read(min(size, READ_CHUNK_BYTES))
                if not part:
                    raise ValueError("incomplete HTTP chunk")
                size -= len(part)
                yield part
            if await reader.readexactly(2) != b"\r\n":
                raise ValueError("invalid HTTP chunk terminator")
    elif length is not None:
        if not length.isdigit():
            raise ValueError("invalid HTTP content length")
        remaining = int(length)
        if remaining > MAX_RESPONSE_BYTES:
            raise ValueError("response exceeds client capture limit")
        while remaining:
            part = await reader.read(min(remaining, READ_CHUNK_BYTES))
            if not part:
                raise ValueError("incomplete HTTP response body")
            remaining -= len(part)
            yield part
    else:
        while part := await reader.read(READ_CHUNK_BYTES):
            yield part


async def exchange(url, body, args, result, started, server):
    target, addresses, tls = server
    writer = None
    last_error = None
    for family, _, _, _, address in addresses:
        try:
            reader, writer = await asyncio.open_connection(
                address[0],
                address[1],
                family=family,
                ssl=tls,
                server_hostname=target.hostname if tls else None,
                limit=MAX_LINE_BYTES,
            )
            break
        except OSError as error:
            last_error = error
    if writer is None:
        raise last_error or OSError("server connection failed")
    completed = False
    try:
        request_target = urllib.parse.quote(
            urllib.parse.urlsplit(url).path, safe="/%:@!$&'()*+,;=-._~"
        )
        if "\r" in target.netloc or "\n" in target.netloc:
            raise ValueError("invalid HTTP host")
        headers = (
            f"POST {request_target} HTTP/1.1\r\nHost: {target.netloc}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        writer.write(headers)
        await writer.drain()
        view = memoryview(body)
        for offset in range(0, len(view), SEND_CHUNK_BYTES):
            writer.write(view[offset : offset + SEND_CHUNK_BYTES])
            result["client_write_buffer_peak_bytes"] = max(
                result["client_write_buffer_peak_bytes"], writer.transport.get_write_buffer_size()
            )
            await writer.drain()
        status, headers = await read_headers(reader)
        result["status"] = status
        capture = StreamCapture(started) if args.stream and status == 200 else None
        data = bytearray()
        async for part in body_parts(reader, headers):
            if capture is not None:
                capture.feed(part)
            else:
                if len(data) + len(part) > MAX_RESPONSE_BYTES:
                    raise ValueError("response exceeds client capture limit")
                data.extend(part)
            if time.perf_counter() - started >= args.timeout:
                raise TimeoutError("request deadline exceeded")
        if status != 200:
            result["outcome"] = "http_error"
            result["error"] = data.decode("utf-8", errors="replace")
        else:
            result.update(
                capture.result() if capture is not None else parse_complete(io.BytesIO(data))
            )
        if time.perf_counter() - started >= args.timeout:
            raise TimeoutError("request deadline exceeded")
        completed = True
    finally:
        if completed:
            writer.close()
        else:
            writer.transport.abort()
        try:
            await asyncio.wait_for(writer.wait_closed(), TRANSPORT_CLOSE_SECONDS)
            result["transport_close_confirmed"] = True
        except asyncio.CancelledError:
            writer.transport.abort()
            try:
                await asyncio.wait_for(writer.wait_closed(), TRANSPORT_CLOSE_SECONDS)
                result["transport_close_confirmed"] = True
            except (OSError, TimeoutError):
                pass
            raise
        except (OSError, TimeoutError) as error:
            writer.transport.abort()
            if completed:
                raise OSError("client transport close was not confirmed") from error


async def send(url, body, fixtures, args, warmup, server):
    start_unix_ns = time.time_ns()
    started = time.perf_counter()
    result = {
        "fixtures": [{"file": f, "sha256": digest} for f, digest, _ in fixtures],
        "warmup": warmup,
        "start_unix_ns": start_unix_ns,
        "status": None,
        "outcome": "transport_error",
        "error": None,
        "client_write_buffer_peak_bytes": 0,
        "transport_close_confirmed": False,
    }
    try:
        await asyncio.wait_for(exchange(url, body, args, result, started, server), args.timeout)
        tokens = (result.get("usage") or {}).get("completion_tokens")
        if result["outcome"] == "http_error":
            pass
        elif args.require_output_tokens is not None and (
            type(tokens) is not int or tokens != args.require_output_tokens
        ):
            result["outcome"] = "output_length_mismatch"
            result["error"] = (
                f"expected {args.require_output_tokens} output tokens, observed {tokens}"
            )
        else:
            result["outcome"] = "success"
    except asyncio.CancelledError:
        result["outcome"] = "cancelled"
        result["error"] = "client cancelled request"
    except TimeoutError:
        result["outcome"] = "timeout"
        result["error"] = "request deadline exceeded"
    except Exception as error:  # Preserve each request's failure without hiding other samples.
        result["outcome"] = "invalid_response" if result["status"] else "transport_error"
        result["error"] = f"{type(error).__name__}: {error}"
    result["latency_ms"] = (time.perf_counter() - started) * 1000
    result["end_unix_ns"] = time.time_ns()
    return result


def percentiles(values):
    values = sorted(values)
    if not values:
        return None

    def nearest_rank(fraction):
        return values[math.ceil(fraction * len(values)) - 1]

    return {
        "count": len(values),
        "p50_ms": statistics.median(values),
        "p95_ms": nearest_rank(0.95),
        "p99_ms": nearest_rank(0.99),
        "max_ms": values[-1],
    }


def valid_latency(value):
    if type(value) is int:
        return value > 0
    return type(value) is float and math.isfinite(value) and value > 0


def comparison_preflight(reference, candidate):
    """Check observable comparability without claiming a performance qualification."""
    if not isinstance(reference, dict) or not isinstance(candidate, dict):
        raise ValueError("comparison reports must be JSON objects")
    issues = []
    if (
        reference.get("capture_contract") != TEXT_CAPTURE_CONTRACT
        or candidate.get("capture_contract") != TEXT_CAPTURE_CONTRACT
    ):
        issues.append("capture_contract_missing_or_mismatched")
    workload = candidate.get("workload")
    try:
        same_workload = isinstance(workload, dict) and json.dumps(
            workload, sort_keys=True, allow_nan=False
        ) == json.dumps(reference.get("workload"), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        same_workload = False
    if not same_workload:
        issues.append("workload_mismatch")
    model = candidate.get("model")
    if not isinstance(model, str) or not model or model != reference.get("model"):
        issues.append("requested_model_mismatch_or_missing")
    left = reference.get("samples")
    right = candidate.get("samples")
    requested = workload.get("samples") if isinstance(workload, dict) else None
    if (
        not isinstance(requested, int)
        or isinstance(requested, bool)
        or requested <= 0
        or not isinstance(left, list)
        or not isinstance(right, list)
        or len(left) != requested
        or len(right) != requested
    ):
        issues.append("sample_count_mismatch")
        left, right = [], []
    expected_warmup = workload.get("warmup") if isinstance(workload, dict) else None
    warmups_valid = True
    for report in (reference, candidate):
        warmups = report.get("warmups")
        if (
            type(expected_warmup) is not int
            or expected_warmup < 0
            or not isinstance(warmups, list)
            or len(warmups) != expected_warmup
            or any(
                not isinstance(item, dict) or item.get("outcome") != "success" for item in warmups
            )
        ):
            issues.append("warmup_incomplete")
            warmups_valid = False
    pairs = [
        (f"sample_{index}", a, b) for index, (a, b) in enumerate(zip(left, right, strict=True))
    ]
    if warmups_valid:
        pairs.extend(
            (f"warmup_{index}", a, b)
            for index, (a, b) in enumerate(
                zip(reference["warmups"], candidate["warmups"], strict=True)
            )
        )
    for prefix, baseline, current in pairs:
        if not isinstance(baseline, dict) or not isinstance(current, dict):
            issues.append(f"{prefix}:invalid_sample")
            continue
        if (
            baseline.get("outcome") != "success"
            or current.get("outcome") != "success"
            or type(baseline.get("status")) is not int
            or type(current.get("status")) is not int
            or baseline.get("status") != 200
            or current.get("status") != 200
        ):
            issues.append(f"{prefix}:request_failed")
        if any(not valid_latency(sample.get("latency_ms")) for sample in (baseline, current)):
            issues.append(f"{prefix}:latency_missing_or_invalid")
        if (
            isinstance(workload, dict)
            and workload.get("stream")
            and any(
                not valid_latency(sample.get("ttft_ms")) or not sample.get("text")
                for sample in (baseline, current)
            )
        ):
            issues.append(f"{prefix}:first_visible_content_missing_or_invalid")
        fixtures = baseline.get("fixtures")
        if (
            not isinstance(fixtures, list)
            or not fixtures
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("file"), str)
                or not item["file"]
                or not isinstance(item.get("sha256"), str)
                or len(item["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in item["sha256"])
                for item in fixtures
            )
            or fixtures != current.get("fixtures")
        ):
            issues.append(f"{prefix}:fixtures_mismatch")
        if not isinstance(baseline.get("text"), str) or baseline.get("text") != current.get("text"):
            issues.append(f"{prefix}:text_mismatch")
        if (
            not isinstance(baseline.get("response_model"), str)
            or not baseline["response_model"]
            or baseline["response_model"] != current.get("response_model")
        ):
            issues.append(f"{prefix}:response_model_mismatch_or_missing")
        if (
            not isinstance(baseline.get("finish_reason"), str)
            or baseline.get("finish_reason") not in TEXT_FINISH_REASONS
            or baseline.get("finish_reason") != current.get("finish_reason")
        ):
            issues.append(f"{prefix}:finish_reason_mismatch")
        for key in ("prompt_tokens", "completion_tokens"):
            baseline_usage = baseline.get("usage")
            current_usage = current.get("usage")
            baseline_tokens = baseline_usage.get(key) if isinstance(baseline_usage, dict) else None
            current_tokens = current_usage.get(key) if isinstance(current_usage, dict) else None
            if (
                type(baseline_tokens) is not int
                or type(current_tokens) is not int
                or baseline_tokens < 0
                or baseline_tokens != current_tokens
            ):
                issues.append(f"{prefix}:{key}_mismatch_or_missing")
    return {
        "passed": not issues,
        "issues": issues,
        "scope": (
            "Observable request/output consistency only; model weights, effective server "
            "settings, hardware, token identities and numerical quality require separate checks. "
            "For URL sources the returned bytes are not verified by the client."
        ),
    }


def telemetry_summary(path, start_unix_ns, end_unix_ns):
    """Summarize the existing hardware monitor in the same client time window."""
    spec = importlib.util.spec_from_file_location(
        "serve_compare", Path(__file__).with_name("serve-compare.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.load_telemetry(path)
    if not data["metadata"]:
        raise ValueError("monitor metadata missing")
    window = [(start_unix_ns / 1e9, end_unix_ns / 1e9)]
    raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rss = []
    for sample in raw:
        at = sample.get("unix_seconds")
        if sample.get("type") != "sample" or at is None or not window[0][0] <= at <= window[0][1]:
            continue
        processes = sample.get("processes") or {}
        if processes:
            rss.append(sum(process.get("rss_bytes", 0) for process in processes.values()))
    return {
        "path": str(path),
        "monitor_pid": data["metadata"].get("pid"),
        "interval_seconds": data["metadata"].get("interval_seconds"),
        "process_cpu": module.process_summary(data["samples"], window),
        "gpu": module.gpu_summary(data["samples"], window),
        "process_tree_rss_peak_bytes": max(rss) if rss else None,
        "rss_note": "Sum of per-process RSS can count shared pages more than once.",
    }


def run(args):
    reference = read_json(args.reference_report) if args.reference_report else None
    if reference is not None and not isinstance(reference, dict):
        raise ValueError("reference report must be a JSON object")
    manifest, manifest_sha256, images = fixture_payloads(args)
    width = args.images_per_request
    groups = [images[index : index + width] for index in range(0, len(images), width)]
    bodies = [request_body(args, group) for group in groups]
    groups = [[(name, digest, None) for name, digest, _ in group] for group in groups]
    del images
    url = args.url.rstrip("/") + "/v1/chat/completions"
    server = resolve_server(url, args.connect_ip)
    return asyncio.run(
        run_async(args, reference, manifest, manifest_sha256, groups, bodies, url, server)
    )


async def run_async(args, reference, manifest, manifest_sha256, groups, bodies, url, server):
    width = args.images_per_request
    warmup = []
    interrupted = False
    stop_reason = None
    for body, group in zip(bodies[: args.warmup], groups[: args.warmup], strict=True):
        item = await send(url, body, group, args, True, server)
        warmup.append(item)
        if item["outcome"] != "success":
            interrupted = item["outcome"] == "cancelled"
            stop_reason = f"warmup:{item['outcome']}"
            break
    start_unix_ns = time.time_ns()
    started = time.perf_counter()
    samples = []
    if not interrupted and all(item["outcome"] == "success" for item in warmup):
        samples = [None] * args.samples
        next_sample = 0
        stopped = False

        async def worker():
            nonlocal next_sample, stopped, stop_reason
            while not stopped:
                index = next_sample
                next_sample += 1
                if index >= args.samples:
                    break
                samples[index] = await send(
                    url,
                    bodies[args.warmup + index],
                    groups[args.warmup + index],
                    args,
                    False,
                    server,
                )
                result = samples[index]
                capacity_rejection = result["outcome"] == "http_error" and result["status"] == 429
                if result["outcome"] not in ("success", "cancelled") and not capacity_rejection:
                    stopped = True
                    stop_reason = f"sample:{result['outcome']}"
                    for task in workers:
                        if task is not asyncio.current_task() and not task.done():
                            task.cancel()
                    break
                if asyncio.current_task().cancelling():
                    break

        workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            interrupted = True
            stop_reason = "client_cancelled"
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
    elapsed = time.perf_counter() - started
    end_unix_ns = time.time_ns()
    samples = [sample for sample in samples if sample is not None]
    successful = [item for item in samples if item["outcome"] == "success"]
    failed = [item for item in samples if item["outcome"] != "success"]
    first_tokens = [item["ttft_ms"] for item in successful if item.get("ttft_ms") is not None]
    token_spacing = [
        item["tpot_estimate_ms"] for item in successful if item.get("tpot_estimate_ms") is not None
    ]
    report = {
        "scope": "closed-loop image-chat client; text HTTP response; not a release gate",
        "capture_contract": TEXT_CAPTURE_CONTRACT,
        "first_token_scope": "first visible content delta" if args.stream else None,
        "concurrency_scope": "client request limit; server-side concurrency is not measured",
        "client_preparation": {
            "fixture_bytes_limit": MAX_FIXTURE_BYTES,
            "conservative_preconstruction_limit": args.max_prepared_mib * MIB_BYTES,
            "retained_request_body_bytes": sum(len(body) for body in bodies),
            "note": (
                "Preconstruction accounts for bounded source/base64, request bodies and "
                "temporary payload estimates; it is not a Python RSS limit. Source strings "
                "are released after request body construction. Concurrent responses, "
                "JSON/SSE objects, transport buffers and telemetry use additional memory."
            ),
        },
        "url": url,
        "model": args.model,
        "engine_label": args.engine_label,
        "workload": {
            "size": args.size,
            "concurrency": args.concurrency,
            "samples": args.samples,
            "warmup": args.warmup,
            "images_per_request": width,
            "image_source": args.image_source,
            "fixture_seed": manifest.get("seed"),
            "fixture_offset": args.offset,
            "manifest_sha256": manifest_sha256,
            "max_tokens": args.max_tokens,
            "required_output_tokens": args.require_output_tokens,
            "processor_options": args.processor_options,
            "prompt": args.prompt,
            "stream": args.stream,
            "timeout_seconds": args.timeout,
        },
        "window": {
            "start_unix_ns": start_unix_ns,
            "end_unix_ns": end_unix_ns,
            "elapsed_seconds_including_drain": elapsed,
        },
        "counts": {
            "requested": args.samples,
            "attempted": len(samples),
            "unattempted": args.samples - len(samples),
            "successes": len(successful),
            "failures": len(failed),
            "by_outcome": dict(collections.Counter(item["outcome"] for item in samples)),
            "by_http_status": dict(collections.Counter(str(item["status"]) for item in samples)),
        },
        "interrupted": interrupted,
        "stop_reason": stop_reason,
        "successes_per_second": len(successful) / elapsed if samples else None,
        "offered_per_second": len(samples) / elapsed if samples else None,
        "latency_success": percentiles([item["latency_ms"] for item in successful]),
        "latency_failure": percentiles([item["latency_ms"] for item in failed]),
        "first_visible_content": percentiles(first_tokens),
        "tpot_estimate": percentiles(token_spacing),
        "tpot_estimate_scope": (
            "Last minus first visible content event, divided by completion tokens minus one. "
            "Requires visible events in at least two reads and at least two completion tokens. "
            "Events may contain "
            "multiple or hidden tokens and may be coalesced by transport, so this is an "
            "estimate and not a tokenizer-verified per-token latency."
            if args.stream
            else None
        ),
        "percentile_note": (
            "Nearest-rank p95/p99; sparse samples are exploratory; failures are separate."
        ),
        "source_note": (
            "URL mode identifies local fixture bytes, but the remote server may return "
            "different bytes."
        ),
        "timeout_note": (
            "The request deadline covers connection, TLS, send and response. No DNS is used; "
            "non-numeric hosts require --connect-ip. Timing is cooperative and can be delayed "
            "by operating-system scheduling, bounded response parsing and transport close. "
            "Chunked client sending is included in latency; do not mix with older client runs."
        ),
        "connection_ip": server[1][0][4][0],
        "warmups": warmup,
        "samples": samples,
    }
    if args.telemetry_file:
        try:
            if not args.telemetry_file.is_file():
                raise FileNotFoundError(args.telemetry_file)
            report["telemetry"] = telemetry_summary(args.telemetry_file, start_unix_ns, end_unix_ns)
        except (OSError, ValueError, TypeError, KeyError) as error:
            report["telemetry"] = {"error": f"{type(error).__name__}: {error}"}
    if reference is not None:
        report["comparison_preflight"] = comparison_preflight(reference, report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--connect-ip", help="numeric server address; URL host remains the HTTP Host and TLS SNI"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-label", required=True)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--size", required=True)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--samples", required=True, type=int)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0, help="image offset in the manifest")
    parser.add_argument("--images-per-request", type=int, default=1)
    parser.add_argument("--image-source", choices=("data", "url"), default="data")
    parser.add_argument(
        "--max-prepared-mib",
        type=int,
        default=DEFAULT_MAX_PREPARED_MIB,
        help="maximum estimated fixture/request preparation MiB, not a process RSS limit",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--require-output-tokens", type=int)
    parser.add_argument("--processor-options", type=json.loads)
    parser.add_argument("--prompt", default="Describe this image in a few words.")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--reference-report", type=Path, help="previous report for observable comparison checks"
    )
    parser.add_argument(
        "--telemetry-file", type=Path, help="JSONL from the existing hardware-monitor.py"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if (
        not 1 <= args.concurrency <= min(args.samples, MAX_CLIENT_CONCURRENCY)
        or args.images_per_request < 1
    ):
        parser.error(
            f"require 1 <= concurrency <= samples and <= {MAX_CLIENT_CONCURRENCY}, "
            "and images-per-request >= 1"
        )
    if (
        args.warmup < 0
        or args.offset < 0
        or args.max_tokens < 1
        or args.max_prepared_mib < 1
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error("invalid warmup, offset, token, preparation or timeout limit")
    if (
        args.require_output_tokens is not None
        and not 0 <= args.require_output_tokens <= args.max_tokens
    ):
        parser.error("required output tokens must be between zero and max-tokens")
    if args.processor_options is not None and not isinstance(args.processor_options, dict):
        parser.error("processor options must be a JSON object")
    parsed_url = urllib.parse.urlsplit(args.url)
    if (
        parsed_url.scheme not in ("http", "https")
        or not parsed_url.hostname
        or parsed_url.username
        or parsed_url.password
        or parsed_url.query
        or parsed_url.fragment
    ):
        parser.error("url must be HTTP(S), with a host and without credentials, query, or fragment")
    try:
        server_address(parsed_url.hostname, args.connect_ip)
    except ValueError as error:
        parser.error(str(error))
    return args


def main():
    args = parse_args()
    if args.output.exists():
        raise SystemExit("output already exists; reports are never overwritten")
    try:
        report = run(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
            file.write("\n")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise SystemExit(f"image load failed: {error}") from error
    summary = {
        key: report[key]
        for key in (
            "counts",
            "successes_per_second",
            "latency_success",
            "latency_failure",
            "first_visible_content",
            "tpot_estimate",
        )
    }
    if "comparison_preflight" in report:
        summary["comparison_preflight"] = report["comparison_preflight"]
    print(json.dumps(summary, ensure_ascii=False))
    if (
        report["counts"]["failures"]
        or report["counts"]["unattempted"]
        or report["interrupted"]
        or "error" in report.get("telemetry", {})
        or not report.get("comparison_preflight", {"passed": True})["passed"]
        or any(item["outcome"] != "success" for item in report["warmups"])
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
