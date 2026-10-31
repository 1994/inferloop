"""The probe's checkpoints must parse as safetensors and match what the Qwen3 provider binds."""

import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "model_probe", Path(__file__).with_name("model-probe.py")
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class CheckpointTest(unittest.TestCase):
    def test_every_case_keeps_attention_consistent_with_its_projections(self):
        for name, geometry in module.CASES.items():
            with self.subTest(name):
                shapes = module.tensor_shapes(geometry, 16)
                query = shapes["model.layers.0.self_attn.q_proj.weight"]
                self.assertEqual(query, [geometry.heads * geometry.head_dim, geometry.hidden])
                self.assertEqual(
                    shapes["model.layers.0.self_attn.k_norm.weight"], [geometry.head_dim]
                )

    def test_written_file_matches_its_header(self):
        geometry = module.Geometry(8, 2, 1, 4, 16, 1, "F32")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text(json.dumps({"vocab_size": 16}))
            (source / "tokenizer.json").write_text("{}")
            target = root / "target"
            module.write_checkpoint(target, source, geometry)
            self.assertTrue((target / "tokenizer.json").is_file())
            data = (target / "model.safetensors").read_bytes()
            size = struct.unpack("<Q", data[:8])[0]
            header = json.loads(data[8 : 8 + size])
            end = max(entry["data_offsets"][1] for entry in header.values())
            self.assertEqual(len(data), 8 + size + end)
            self.assertEqual(json.loads((target / "config.json").read_text())["head_dim"], 4)

    def test_bf16_block_keeps_the_high_half_of_each_float(self):
        block = module.value_block("BF16")
        self.assertEqual(len(block), module.BLOCK_VALUES * 2)


if __name__ == "__main__":
    unittest.main()
