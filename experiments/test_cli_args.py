import unittest
from local_speculative_runtime.cli import resolve_candidate_json, GGUF_CANDIDATE_PRESETS

class TestCLIArgs(unittest.TestCase):
    def test_mlx_no_candidate_args(self):
        # Should succeed and return default candidate path
        res = resolve_candidate_json("mlx", None, None)
        self.assertEqual(res, "experiments/template_candidates.json")
        
    def test_mlx_with_model_type_fails(self):
        # Specifying model_type with MLX should fail
        with self.assertRaises(ValueError) as ctx:
            resolve_candidate_json("mlx", "qwen", None)
        self.assertIn("--model-type cannot be specified", str(ctx.exception))
        
    def test_mlx_with_candidate_json_fails(self):
        # Specifying candidate_json with MLX should fail
        with self.assertRaises(ValueError) as ctx:
            resolve_candidate_json("mlx", None, "custom.json")
        self.assertIn("--candidate-json cannot be specified", str(ctx.exception))
        
    def test_gguf_requires_candidate_args(self):
        # GGUF must have either model_type or candidate_json
        with self.assertRaises(ValueError) as ctx:
            resolve_candidate_json("llama_cpp", None, None)
        self.assertIn("Either --model-type or --candidate-json must be specified", str(ctx.exception))
        
    def test_gguf_mutually_exclusive_args(self):
        # Cannot specify both
        with self.assertRaises(ValueError) as ctx:
            resolve_candidate_json("gguf", "qwen", "custom.json")
        self.assertIn("Cannot specify both --model-type and --candidate-json", str(ctx.exception))
        
    def test_gguf_resolves_preset(self):
        # model-type qwen should resolve to the preset path
        res = resolve_candidate_json("llama_cpp", "qwen", None)
        self.assertEqual(res, GGUF_CANDIDATE_PRESETS["qwen"])
        
    def test_gguf_unknown_preset_fails(self):
        # Unknown model-type should fail
        with self.assertRaises(ValueError) as ctx:
            resolve_candidate_json("llama_cpp", "unknown", None)
        self.assertIn("Unknown model-type 'unknown'", str(ctx.exception))
        
    def test_gguf_resolves_explicit_json(self):
        # Explicit json path
        res = resolve_candidate_json("gguf", None, "my_custom_candidates.json")
        self.assertEqual(res, "my_custom_candidates.json")

if __name__ == "__main__":
    unittest.main()
