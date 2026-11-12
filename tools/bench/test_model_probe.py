"""The probe's checkpoints must parse as safetensors and match what each provider binds."""

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
                norm = shapes.get("model.layers.0.self_attn.k_norm.weight")
                # Only Qwen3 normalizes queries and keys per head.
                expected = [geometry.head_dim] if geometry.family == "qwen3" else None
                self.assertEqual(norm, expected)
                self.assertEqual("lm_head.weight" in shapes, not geometry.tied)

    def test_a_family_names_its_provider_in_the_configuration(self):
        geometry = module.CASES["llama-untied"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps({"vocab_size": 16, "model_type": "qwen3"})
            )
            target = root / "target"
            module.write_checkpoint(target, source, geometry)
            config = json.loads((target / "config.json").read_text())
            self.assertEqual(config["model_type"], "llama")
            self.assertEqual(config["architectures"], ["LlamaForCausalLM"])
            self.assertFalse(config["tie_word_embeddings"])

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


class TwinTest(unittest.TestCase):
    def test_each_rotary_half_keeps_its_lanes_before_its_padding(self):
        self.assertEqual(module.lanes(6, 8), [0, 1, 2, None, 3, 4, 5, None])
        self.assertEqual(module.padded_width(96), 128)
        self.assertEqual(module.padded_width(100), 128)

    def test_a_twin_pads_projection_rows_and_output_columns(self):
        geometry = module.CASES["llama-96"]
        block = module.value_block("BF16")
        query = module.twin_values("x.q_proj.weight", None, geometry, 128, block)
        output = module.twin_values("x.o_proj.weight", None, geometry, 128, block)
        self.assertEqual(len(query), geometry.heads * 128 * geometry.hidden)
        self.assertEqual(len(output), geometry.hidden * geometry.heads * 128)
        # Padding lanes 48..63 of head 0 are zero rows.
        self.assertTrue(all(v == 0.0 for v in query[48 * geometry.hidden : 64 * geometry.hidden]))
        self.assertIsNone(module.twin_values("x.mlp.up_proj.weight", None, geometry, 128, block))
