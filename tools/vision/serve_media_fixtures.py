#!/usr/bin/env python3
"""Freeze the deterministic image/video fixtures used by the serving media acceptance.

The raw RGB bytes below match `cuda-multimodal-generate` and `cuda-multimodal-media`
exactly. The video is written with the lossless FFV1 codec rather than a lossy MP4, and
every decoded frame is compared byte-for-byte with the source frame before the manifest is
written. The manifest records the source bytes, decoded bytes, ffprobe metadata and the
exact prompts; it does not claim a GPU result.

Run:

    python3 tools/vision/serve_media_fixtures.py --out /tmp/issue59-fixtures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess

HEIGHT = 60
WIDTH = 100
FRAMES = 8
FPS = 30.0
IMAGE_PROMPT = "<|vision_start|><|image_pad|><|vision_end|>Describe this image in a few words."
VIDEO_PROMPT = "<|vision_start|><|video_pad|><|vision_end|>Describe the motion in this clip."


def image(kind: int) -> bytes:
    pixels = bytearray(HEIGHT * WIDTH * 3)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            slot = (y * WIDTH + x) * 3
            vertical = y * 255 // HEIGHT
            horizontal = x * 255 // WIDTH
            if kind == 0:
                pixels[slot : slot + 3] = bytes((vertical, horizontal, 128))
            else:
                pixels[slot : slot + 3] = bytes((horizontal, 255 - vertical, 64))
    return bytes(pixels)


def video() -> list[bytes]:
    frames = []
    for frame in range(FRAMES):
        pixels = bytearray(HEIGHT * WIDTH * 3)
        for y in range(HEIGHT):
            for x in range(WIDTH):
                slot = (y * WIDTH + x) * 3
                stripe = 200 if abs(x - frame * WIDTH // FRAMES) < 10 else 40
                pixels[slot : slot + 3] = bytes(
                    (
                        (y * 255 // HEIGHT + frame * 20) % 256,
                        stripe,
                        (x * 255 // WIDTH + frame * 10) % 256,
                    )
                )
        frames.append(bytes(pixels))
    return frames


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run(arguments: list[str], stdin: bytes | None = None) -> bytes:
    result = subprocess.run(arguments, input=stdin, capture_output=True, check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"command failed ({result.returncode}): {' '.join(arguments)}\n"
            f"{result.stderr.decode(errors='replace')[:2000]}"
        )
    return result.stdout


def encode_png(raw: bytes, path: pathlib.Path) -> None:
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-i",
            "pipe:0",
            "-frames:v",
            "1",
            str(path),
        ],
        raw,
    )


def encode_video(frames: list[bytes], path: pathlib.Path) -> None:
    raw = b"".join(frames)
    run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            str(FPS),
            "-i",
            "pipe:0",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "gbrp",
            "-an",
            str(path),
        ],
        raw,
    )


def decode_frames(path: pathlib.Path, count: int) -> list[bytes]:
    raw = run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            str(count),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
    )
    frame_bytes = WIDTH * HEIGHT * 3
    if len(raw) != frame_bytes * count:
        raise SystemExit(f"decoded video has {len(raw)} bytes, expected {frame_bytes * count}")
    return [raw[index * frame_bytes : (index + 1) * frame_bytes] for index in range(count)]


def probe(path: pathlib.Path) -> dict:
    output = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    stream = json.loads(output)["streams"][0]
    return {
        "width": stream["width"],
        "height": stream["height"],
        "avg_frame_rate": stream["avg_frame_rate"],
        "nb_frames": stream.get("nb_frames"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.out
    root.mkdir(parents=True, exist_ok=True)

    images = {}
    for kind in range(2):
        raw = image(kind)
        path = root / f"image-{kind}.png"
        encode_png(raw, path)
        decoded = run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-i",
                str(path),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ]
        )
        if decoded != raw:
            raise SystemExit(f"image {kind} decode mismatch")
        images[str(kind)] = {
            "path": str(path),
            "width": WIDTH,
            "height": HEIGHT,
            "source_sha256": digest(raw),
            "decoded_sha256": digest(decoded),
            "identity": decoded == raw,
            "prompt": IMAGE_PROMPT,
        }

    frames = video()
    clip = root / "clip-ffv1.mkv"
    encode_video(frames, clip)
    decoded = decode_frames(clip, FRAMES)
    if decoded != frames:
        raise SystemExit("video decode mismatch")
    video_entry = {
        "path": str(clip),
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "frame_count": FRAMES,
        "codec": "ffv1",
        "frame_sha256": [digest(frame) for frame in frames],
        "decoded_frame_sha256": [digest(frame) for frame in decoded],
        "decoded_identity": decoded == frames,
        "ffprobe": probe(clip),
        "prompt": VIDEO_PROMPT,
    }

    manifest = {
        "schema": "issue59-serving-media-fixtures-v1",
        "source": (
            "deterministic raw RGB matching cuda-multimodal-generate and cuda-multimodal-media"
        ),
        "images": images,
        "video": video_entry,
        "note": (
            "Lossless FFV1 is used instead of a lossy MP4; "
            "all decoded frames are byte-identical to the source frames."
        ),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"out": str(root), "manifest": str(root / "manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
