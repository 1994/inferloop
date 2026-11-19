#!/usr/bin/env python3
"""End-to-end reference run for parity against the Rust pipeline.

Runs the official `Qwen3VLForConditionalGeneration` on one deterministic image and prompt and dumps
everything the Rust pipeline needs to reproduce the same step: the source pixels, the rendered
prompt text, the expanded token ids, the final prompt logits and a greedy continuation.

The released configuration runs with DeepStack: the tower's intermediate features are merged and
added to the image tokens after the first decoder layers, and the dump records those features per
level. `--no-deepstack` disables them on this side only, for diagnosing the shared path
(preprocess -> vision tower -> merger -> placeholder merge -> text decoding). The comparison is only
meaningful when the Rust run uses the same package and the same image. F32 with TF32 disabled is the default oracle;
`--dtype bfloat16` is an explicit alternative precision diagnostic.

Run:

    uv run --with "transformers>=5,<6" --with torch --with torchvision --with numpy \
        --with safetensors --with accelerate python3 tools/vision/qwen3vl_2b_parity_reference.py \
        --package /home/r/models/qwen3vl-2b \
        --out artifacts/vision/qwen3vl-2b-parity.safetensors
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# Must match `synthetic_image(0)` in the Rust example byte for byte.
HEIGHT = 60
WIDTH = 100
PROMPT = "Describe this image in a few words."


def synthetic_image() -> np.ndarray:
    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for y in range(HEIGHT):
        for x in range(WIDTH):
            pixels[y, x] = (y * 255 // HEIGHT, x * 255 // WIDTH, 128)
    return pixels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument(
        "--no-deepstack",
        action="store_true",
        help="disable DeepStack injection: the shared-path diagnostic, not the released model",
    )
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.package)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.package, dtype=getattr(torch, args.dtype), device_map="cuda"
    )
    if args.no_deepstack:
        if hasattr(model.config, "vision_config"):
            model.config.vision_config.deepstack_visual_indexes = []
        if hasattr(model.config, "deepstack_visual_indexes"):
            model.config.deepstack_visual_indexes = []
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.eval()

    image = synthetic_image()
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
    ids = inputs["input_ids"][0]
    image_token = int(model.config.image_token_id)
    positions = (ids == image_token).nonzero().flatten().tolist()
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise SystemExit("image tokens are not contiguous; the merge contract changed")

    captured: dict[str, torch.Tensor] = {}
    handles = []
    visual = getattr(model, "visual", None) or getattr(model.model, "visual", None)
    if visual is None:
        raise SystemExit("the model has no visual tower")
    if args.no_deepstack:
        # The model copies this list during construction. Changing config alone leaves
        # DeepStack active and makes the supposedly shared-path oracle incomparable.
        visual.deepstack_visual_indexes = []

    def capture_tower(_module, _inputs, output):
        captured["vision_embeddings"] = output.pooler_output.detach().float()
        features = output.deepstack_features or []
        if args.no_deepstack and features:
            raise RuntimeError("DeepStack must be disabled for shared-path parity")
        if features:
            captured["deepstack_features"] = torch.stack(
                [feature.detach().float() for feature in features]
            )

    handles.append(visual.register_forward_hook(capture_tower))
    with torch.no_grad():
        logits = model(**inputs, use_cache=True).logits[0, -1].float()
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    new_tokens = generated[0][ids.shape[0] :].to(torch.int64)
    for handle in handles:
        handle.remove()
    if "vision_embeddings" not in captured:
        raise SystemExit("the visual tower output was not captured")
    vision_embeddings = captured["vision_embeddings"].reshape(
        -1, model.config.text_config.hidden_size
    )

    tensors = {
        "image": torch.from_numpy(image.reshape(-1).copy()),
        "image_shape": torch.tensor([HEIGHT, WIDTH], dtype=torch.int64),
        "text": torch.frombuffer(bytearray(text.encode("utf-8")), dtype=torch.uint8).clone(),
        "input_ids": ids.to(torch.int64),
        "logits": logits,
        "generated": new_tokens,
        "vision_embeddings": vision_embeddings.contiguous(),
        # The processor's own pixels, so the Rust side can separate tower error from
        # preprocessing error by feeding the tower the identical input.
        "pixel_values": inputs["pixel_values"].float().cpu().contiguous(),
        "grid_thw": inputs["image_grid_thw"].to(torch.int64).reshape(-1),
    }
    if "deepstack_features" in captured:
        tensors["deepstack_features"] = (
            captured["deepstack_features"]
            .reshape(-1, vision_embeddings.shape[0], vision_embeddings.shape[1])
            .contiguous()
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    meta = {
        "model": str(args.package),
        "dtype": args.dtype,
        "tf32": False,
        "deepstack": not args.no_deepstack,
        "image_token_id": image_token,
        "vision_start_token_id": int(model.config.vision_start_token_id),
        "grid_thw": inputs["image_grid_thw"].tolist(),
        "prompt_tokens": int(ids.shape[0]),
        "image_token_span": [positions[0], positions[-1] + 1],
        "merged_visual_tokens": len(positions),
        "generated_tokens": new_tokens.tolist(),
        "generated_text": processor.decode(new_tokens.tolist(), skip_special_tokens=True),
        "deepstack": "disabled on both sides",
        "max_new_tokens": args.max_new_tokens,
        "vision_embeddings_shape": list(vision_embeddings.shape),
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps(meta, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
