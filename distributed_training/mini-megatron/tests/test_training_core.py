import unittest


class FakeModel:
    def __init__(self) -> None:
        self.parameters_called = False

    def parameters(self):
        self.parameters_called = True
        return []


class FakeOptimizer:
    def __init__(self) -> None:
        self.zero_grad_calls = 0
        self.step_calls = 0
        self.param_groups = [{"lr": 0.0}]

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


class FakeScheduler:
    def __init__(self) -> None:
        self.increments: list[int] = []

    def step(self, *, increment: int) -> None:
        self.increments.append(increment)


class FakeLoss:
    def __init__(self, value: float, backward_values: list[float]) -> None:
        self.value = value
        self.backward_values = backward_values

    def __truediv__(self, divisor: int):
        return FakeLoss(self.value / divisor, self.backward_values)

    def backward(self) -> None:
        self.backward_values.append(self.value)


class TrainingCoreTest(unittest.TestCase):
    def _cfg(self):
        from mini_megatron.config import ConfigContainer, DataConfig, Qwen36ModelConfig, TrainConfig

        return ConfigContainer(
            model=Qwen36ModelConfig(
                vocab_size=128,
                hidden_size=32,
                intermediate_size=64,
                num_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_experts=4,
                num_experts_per_tok=2,
                max_position_embeddings=8,
            ),
            train=TrainConfig(train_iters=2, micro_batch_size=2, global_batch_size=8),
            data=DataConfig(seq_length=8, vocab_size=128),
        )

    def test_setup_model_and_optimizer_calls_megatron_lm_style_providers(self):
        from mini_megatron.distributed import DistributedContext
        from mini_megatron.training_core import setup_model_and_optimizer

        cfg = self._cfg()
        ctx = DistributedContext(rank=0, local_rank=0, world_size=2, device="cpu")
        events: list[str] = []

        def model_provider(_cfg, _ctx):
            events.append("model")
            return FakeModel()

        def optimizer_provider(_cfg, model):
            events.append("optimizer")
            return FakeOptimizer()

        def scheduler_provider(_cfg, optimizer):
            events.append("scheduler")
            return FakeScheduler()

        def data_iterator_provider(_cfg, _ctx):
            events.append("data")
            return iter([{"tokens": 1}])

        setup = setup_model_and_optimizer(
            cfg,
            ctx,
            model_provider=model_provider,
            optimizer_provider=optimizer_provider,
            scheduler_provider=scheduler_provider,
            data_iterator_provider=data_iterator_provider,
        )

        self.assertEqual(events, ["model", "optimizer", "scheduler", "data"])
        self.assertEqual(setup.num_microbatches, 2)
        self.assertEqual(setup.world_size, 2)

    def test_forward_backward_no_pipeline_scales_each_microbatch_loss(self):
        from mini_megatron.training_core import forward_backward_no_pipeline

        backward_values: list[float] = []

        def forward_step(batch, _model):
            loss = FakeLoss(float(batch), backward_values)
            return loss, {"lm loss": float(batch)}

        metrics = forward_backward_no_pipeline(
            forward_step_func=forward_step,
            data_iterator=iter([1, 2]),
            model=FakeModel(),
            num_microbatches=2,
        )

        self.assertEqual(backward_values, [0.5, 1.0])
        self.assertEqual(metrics, [{"lm loss": 1.0}, {"lm loss": 2.0}])

    def test_train_step_updates_optimizer_scheduler_and_returns_mean_loss(self):
        from mini_megatron.training_core import TrainingSetup, train_step

        cfg = self._cfg()
        optimizer = FakeOptimizer()
        scheduler = FakeScheduler()
        setup = TrainingSetup(
            model=FakeModel(),
            optimizer=optimizer,
            scheduler=scheduler,
            data_iterator=iter([]),
            num_microbatches=2,
            world_size=2,
        )

        def fake_forward_backward(**kwargs):
            self.assertEqual(kwargs["num_microbatches"], 2)
            return [{"lm loss": 1.0}, {"lm loss": 3.0}]

        result = train_step(
            lambda _batch, _model: (None, {}),
            setup,
            cfg,
            forward_backward_func=fake_forward_backward,
        )

        self.assertEqual(optimizer.zero_grad_calls, 1)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertEqual(scheduler.increments, [8])
        self.assertEqual(result.loss_dict, {"lm loss": 2.0})
        self.assertEqual(result.consumed_samples, 8)


if __name__ == "__main__":
    unittest.main()
