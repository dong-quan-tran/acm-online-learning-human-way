
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")

from src import plot_results


class TestPlotResults(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        root = Path(self.temp_dir.name)
        self.results_path = root / "online_results.csv"
        self.figures_path = root / "figures"

        self.path_patch = patch.multiple(
            plot_results,
            RESULTS_PATH=self.results_path,
            FIGURES_PATH=self.figures_path,
        )
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def write_csv(self, rows):
        with open(self.results_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "mode", "run_id", "examples_processed",
                "accuracy", "eval_split"
            ])
            writer.writerows(rows)

    def test_valid_data_generates_plot(self):
        self.write_csv([
            ["online", "test_run", 0, 0.01, "validation"],
            ["online", "test_run", 1000, 0.08, "validation"],
        ])

        plot_results.plot_learning_curve()

        self.assertTrue(
            (self.figures_path / "online_learning_curve.png").exists()
        )

    def test_incomplete_row_is_skipped(self):
        self.write_csv([
            ["online", "test_run", 1000, 0.08, "validation"],
            ["online", "test_run", 2000],
        ])

        plot_results.plot_learning_curve()

        self.assertTrue(
            (self.figures_path / "online_learning_curve.png").exists()
        )

    def test_invalid_accuracy_is_skipped(self):
        self.write_csv([
            ["online", "test_run", 1000, 0.08, "validation"],
            ["online", "test_run", 2000, 1.5, "validation"],
        ])

        plot_results.plot_learning_curve()

        self.assertTrue(
            (self.figures_path / "online_learning_curve.png").exists()
        )

    def test_missing_file(self):
        plot_results.plot_learning_curve()

        self.assertFalse(self.figures_path.exists())

    def test_empty_csv(self):
        self.results_path.touch()

        plot_results.plot_learning_curve()

        self.assertFalse(self.figures_path.exists())

    def test_no_valid_rows(self):
        self.write_csv([
            ["online", "test_run", -100, 0.08, "validation"],
            ["online", "test_run", 1000, 2.0, "validation"],
        ])

        plot_results.plot_learning_curve()

        self.assertFalse(self.figures_path.exists())


if __name__ == "__main__":
    unittest.main()
