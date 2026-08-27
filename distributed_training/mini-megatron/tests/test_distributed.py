import os
import unittest
from unittest.mock import patch


class DistributedEnvTest(unittest.TestCase):
    def test_resolves_torchrun_environment(self):
        from mini_megatron.distributed import resolve_distributed_context

        env = {
            "RANK": "3",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "8",
        }
        with patch.dict(os.environ, env, clear=True):
            ctx = resolve_distributed_context()

        self.assertEqual(ctx.rank, 3)
        self.assertEqual(ctx.local_rank, 1)
        self.assertEqual(ctx.world_size, 8)
        self.assertTrue(ctx.is_distributed)

    def test_resolves_slurm_environment(self):
        from mini_megatron.distributed import resolve_distributed_context

        env = {
            "SLURM_PROCID": "5",
            "SLURM_LOCALID": "1",
            "SLURM_NTASKS": "16",
        }
        with patch.dict(os.environ, env, clear=True):
            ctx = resolve_distributed_context()

        self.assertEqual(ctx.rank, 5)
        self.assertEqual(ctx.local_rank, 1)
        self.assertEqual(ctx.world_size, 16)
        self.assertTrue(ctx.is_distributed)


if __name__ == "__main__":
    unittest.main()
