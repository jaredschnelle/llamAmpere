"""Standalone Qwen4Exp draft conversion must retain shared output tensors."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "gguf-py")]
from conversion.qwen4exp import Qwen4ExpTextModel


class MtpExportFilterTests(unittest.TestCase):
    def setUp(self):
        self.saved = {key: getattr(Qwen4ExpTextModel, key) for key in
                      ("_original_block_count", "mtp_only", "no_mtp", "opt_num_mtp_layers")}
        Qwen4ExpTextModel._original_block_count = 48
        Qwen4ExpTextModel.mtp_only = True
        Qwen4ExpTextModel.no_mtp = False

    def tearDown(self):
        for key, value in self.saved.items():
            setattr(Qwen4ExpTextModel, key, value)

    def filtered(self, name):
        item = Qwen4ExpTextModel.filter_tensors((name, lambda: None))
        return item[0] if item else None

    def test_shared_output_tensors_survive(self):
        for suffix in ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight"):
            for prefix in ("model.", "model.language_model."):
                with self.subTest(prefix=prefix, suffix=suffix):
                    name = "hyper_connection_mixer." + suffix
                    self.assertEqual(self.filtered(prefix + name), "model." + name)
        self.assertEqual(self.filtered("model.language_model.embed_tokens.weight"), "model.embed_tokens.weight")
        self.assertEqual(self.filtered("lm_head.weight"), "lm_head.weight")

    def test_draft_remapping_and_exclusion(self):
        self.assertEqual(self.filtered("mtp.layers.0.self_attn.q_proj.weight"),
                         "model.layers.48.self_attn.q_proj.weight")
        self.assertEqual(self.filtered("model.language_model.mtp.hyper_connection_mixer.hc_norm.weight"),
                         "model.layers.48.hyper_connection_mixer.hc_norm.weight")
        self.assertIsNone(self.filtered("model.language_model.layers.0.self_attn.q_proj.weight"))
        self.assertIsNone(self.filtered("model.visual.patch_embed.proj.weight"))

    def test_no_mtp_excludes_draft_mixer(self):
        Qwen4ExpTextModel.mtp_only = False
        Qwen4ExpTextModel.no_mtp = True
        self.assertIsNone(self.filtered("model.language_model.mtp.hyper_connection_mixer.hc_norm.weight"))


if __name__ == "__main__":
    unittest.main()
