import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.prepare_real_dataset import split_sources, validation_count


class PrepareRealDatasetTests(unittest.TestCase):
    def test_validation_count_keeps_one_example_for_train(self):
        self.assertEqual(validation_count(2, 0.5), 1)
        self.assertEqual(validation_count(1, 0.5), 0)

    def test_split_sources_is_deterministic(self):
        paths = [Path(f"sample_{index}.jpg") for index in range(10)]
        train_a, val_a = split_sources(paths, val_ratio=0.2, seed=42)
        train_b, val_b = split_sources(paths, val_ratio=0.2, seed=42)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(len(val_a), 2)
        self.assertEqual(len(train_a), 8)


if __name__ == "__main__":
    unittest.main()
