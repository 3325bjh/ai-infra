import unittest


try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for model tests")
class ModelingTest(unittest.TestCase):
    def test_tiny_qwen36_forward_computes_loss(self):
        from mini_megatron.config import Qwen36ModelConfig
        from mini_megatron.modeling_qwen36 import Qwen36ForCausalLM

        cfg = Qwen36ModelConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=8,
            dropout=0.0,
        )
        model = Qwen36ForCausalLM(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = input_ids.roll(shifts=-1, dims=1)

        output = model(input_ids=input_ids, labels=labels)

        self.assertEqual(tuple(output.logits.shape), (2, 8, cfg.vocab_size))
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))

    def test_switch_aux_loss_is_added_to_language_model_loss(self):
        from mini_megatron.config import Qwen36ModelConfig
        from mini_megatron.modeling_qwen36 import Qwen36ForCausalLM

        cfg = Qwen36ModelConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=8,
            router_aux_loss_coef=0.01,
        )
        model = Qwen36ForCausalLM(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = input_ids.roll(shifts=-1, dims=1)

        output = model(input_ids=input_ids, labels=labels)

        self.assertIsNotNone(output.loss)
        self.assertIsNotNone(output.aux_loss)
        self.assertGreaterEqual(output.aux_loss.item(), 0.0)
        self.assertTrue(torch.isfinite(output.aux_loss))


if __name__ == "__main__":
    unittest.main()
