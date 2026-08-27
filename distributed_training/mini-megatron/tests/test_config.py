import unittest


class ConfigTest(unittest.TestCase):
    def test_qwen36_tiny_recipe_has_megatron_bridge_training_defaults(self):
        from mini_megatron.recipes import qwen36_35b_a3b_tiny_config

        cfg = qwen36_35b_a3b_tiny_config()

        self.assertEqual(cfg.model.hf_model_id, "Qwen/Qwen3.6-35B-A3B")
        self.assertEqual(cfg.model.num_experts, 8)
        self.assertEqual(cfg.model.num_experts_per_tok, 2)
        self.assertEqual(cfg.model.num_attention_heads % cfg.model.num_key_value_heads, 0)
        self.assertEqual(cfg.train.micro_batch_size, 2)
        self.assertEqual(cfg.data.seq_length, cfg.model.max_position_embeddings)

    def test_gradient_accumulation_uses_global_batch_micro_batch_and_world_size(self):
        from mini_megatron.config import (
            ConfigContainer,
            DataConfig,
            Qwen36ModelConfig,
            TrainConfig,
        )

        cfg = ConfigContainer(
            model=Qwen36ModelConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_experts=4,
                num_experts_per_tok=2,
                max_position_embeddings=16,
            ),
            train=TrainConfig(train_iters=2, micro_batch_size=2, global_batch_size=16),
            data=DataConfig(seq_length=16, vocab_size=128),
        )

        self.assertEqual(cfg.gradient_accumulation_steps(world_size=4), 2)

    def test_invalid_gqa_shape_is_rejected(self):
        from mini_megatron.config import Qwen36ModelConfig

        with self.assertRaises(ValueError):
            Qwen36ModelConfig(
                vocab_size=128,
                hidden_size=30,
                intermediate_size=64,
                num_layers=1,
                num_attention_heads=8,
                num_key_value_heads=3,
                num_experts=4,
                num_experts_per_tok=2,
                max_position_embeddings=16,
            )


if __name__ == "__main__":
    unittest.main()
