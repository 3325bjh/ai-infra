import unittest


try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for data tests")
class DataTest(unittest.TestCase):
    def test_mock_dataset_returns_shifted_language_model_batch(self):
        from mini_megatron.data import MockTokenDataset, collate_lm_batch

        dataset = MockTokenDataset(vocab_size=128, seq_length=8, num_samples=4, seed=1234)
        sample = dataset[0]
        batch = collate_lm_batch([sample, dataset[1]])

        self.assertEqual(tuple(sample["input_ids"].shape), (8,))
        self.assertEqual(tuple(sample["labels"].shape), (8,))
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 8))
        self.assertEqual(tuple(batch["labels"].shape), (2, 8))
        self.assertTrue(torch.equal(sample["input_ids"][1:], sample["labels"][:-1]))


if __name__ == "__main__":
    unittest.main()
