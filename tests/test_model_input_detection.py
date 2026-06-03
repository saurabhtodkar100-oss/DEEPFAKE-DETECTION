import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis_engine import model_uses_raw_255_inputs


class Dense:
    def __init__(self, layers=None):
        self.layers = layers or []


class Functional:
    def __init__(self, layers=None):
        self.layers = layers or []


class Rescaling:
    def __init__(self, layers=None):
        self.layers = layers or []


class InputScaleDetectionTests(unittest.TestCase):
    def test_detects_nested_rescaling_layer(self):
        model = Functional(layers=[Dense(), Functional(layers=[Rescaling()])])
        self.assertTrue(model_uses_raw_255_inputs(model))

    def test_returns_false_when_rescaling_is_absent(self):
        model = Functional(layers=[Dense(), Functional(layers=[Dense()])])
        self.assertFalse(model_uses_raw_255_inputs(model))


if __name__ == "__main__":
    unittest.main()
