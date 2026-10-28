
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from sklearn.naive_bayes import GaussianNB

# Allow tests to import files from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_gaussiannb import flatten_images, evaluate


class TestGaussianNB(unittest.TestCase):

    def test_flatten_images(self):
        """Images should become flattened feature vectors."""
        images = torch.rand(4, 3, 32, 32)

        features = flatten_images(images)

        self.assertEqual(features.shape, (4, 3072))
        self.assertTrue(np.all(features >= 0))
        self.assertTrue(np.all(features <= 1))

    def test_first_partial_fit(self):
        """The first update should accept the full class list."""
        model = GaussianNB()

        X = np.array([
            [0.1, 0.2],
            [0.2, 0.3],
            [0.8, 0.7],
            [0.9, 0.8],
        ])
        y = np.array([0, 0, 1, 1])

        model.partial_fit(X, y, classes=np.array([0, 1]))

        self.assertEqual(model.classes_.tolist(), [0, 1])

    def test_multiple_updates(self):
        """The same model should learn from multiple batches."""
        model = GaussianNB()

        X1 = np.array([[0.1, 0.2], [0.8, 0.9]])
        y1 = np.array([0, 1])

        X2 = np.array([[0.2, 0.3], [0.9, 0.8]])
        y2 = np.array([0, 1])

        model.partial_fit(X1, y1, classes=np.array([0, 1]))
        model.partial_fit(X2, y2)

        self.assertEqual(model.class_count_.sum(), 4)

    def test_valid_predictions(self):
        """Predictions should contain valid class labels."""
        model = GaussianNB()

        X = np.array([
            [0.1, 0.2],
            [0.2, 0.3],
            [0.8, 0.7],
            [0.9, 0.8],
        ])
        y = np.array([0, 0, 1, 1])

        model.partial_fit(X, y, classes=np.array([0, 1]))
        predictions = model.predict(X)

        self.assertEqual(len(predictions), 4)
        self.assertTrue(np.isin(predictions, [0, 1]).all())

    def test_evaluate_accuracy(self):
        """Evaluation should calculate accuracy correctly."""
        model = GaussianNB()

        X = np.array([
            [0.1, 0.2],
            [0.2, 0.3],
            [0.8, 0.7],
            [0.9, 0.8],
        ])
        y = np.array([0, 0, 1, 1])

        model.partial_fit(X, y, classes=np.array([0, 1]))

        images = torch.tensor(X, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.long)
        sample_ids = ["a", "b", "c", "d"]

        loader = [(images, labels, sample_ids)]

        with patch("train_gaussiannb.flatten_images",
                   side_effect=lambda x: x.numpy()):
            accuracy = evaluate(model, loader)

        expected = np.mean(model.predict(X) == y)
        self.assertAlmostEqual(accuracy, expected)


if __name__ == "__main__":
    unittest.main()
