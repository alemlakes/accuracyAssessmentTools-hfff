import numpy as np
import pandas as pd

from acc_assessment.stehman import Stehman
from acc_assessment.utils import AccuracyAssessment


class MCEM(AccuracyAssessment):
    """
    Monte Carlo Error Matrix estimator.

    Draws crisp map/reference class realizations from per-point class
    probabilities and propagates uncertainty through a crisp design-based
    estimator (default: Stehman).
    """

    def __init__(
        self,
        map_data,
        ref_data,
        strata_col,
        id_col,
        strata_population,
        n_simulations=10000,
        estimator_cls=Stehman,
        random_state=None,
    ):
        assert np.all(map_data.shape == ref_data.shape)
        assert (map_data[id_col] == ref_data[id_col]).all()
        assert (map_data.columns == ref_data.columns).all()

        self.strata_col = strata_col
        self.id_col = id_col
        self.n_simulations = int(n_simulations)
        self.estimator_cls = estimator_cls
        self.random_state = random_state

        self.strata_population = {
            k: v for k, v in iter(strata_population.items())
            if k in np.unique(map_data[strata_col].values)
        }

        self.strata_classes = map_data[strata_col].values
        self.ids = map_data[id_col].values

        self.all_classes = np.array(
            [x for x in map_data.columns if x not in [strata_col, id_col]]
        )

        self.map_probs = map_data[self.all_classes].astype(float).reset_index(drop=True)
        self.ref_probs = ref_data[self.all_classes].astype(float).reset_index(drop=True)

        self.N = np.sum(list(self.strata_population.values()))

        self._simulations_ran = False
        self._results = None

    def _normalize_probabilities(self, probs):
        probs = probs.astype(float)
        row_sums = probs.sum(axis=1)
        if np.any(row_sums <= 0):
            msg = "Each row of class probabilities must have a positive sum"
            raise ValueError(msg)
        return probs / row_sums[:, None]

    def _sample_classes(self, probs, rng):
        probs = self._normalize_probabilities(probs)
        cdf = np.cumsum(probs, axis=1)
        draws = rng.random(size=probs.shape[0])
        sampled_indices = np.sum(draws[:, None] > cdf, axis=1)
        return self.all_classes[sampled_indices]

    def _empty_results(self):
        return {
            "overall": np.zeros(self.n_simulations),
            "users": {k: np.zeros(self.n_simulations) for k in self.all_classes},
            "producers": {k: np.zeros(self.n_simulations) for k in self.all_classes},
            "area": {k: np.zeros(self.n_simulations) for k in self.all_classes},
            "error_matrix": [],
        }

    def _run_simulations(self):
        rng = np.random.default_rng(self.random_state)
        results = self._empty_results()

        for sim_idx in range(self.n_simulations):
            sampled_map = self._sample_classes(self.map_probs.values, rng)
            sampled_ref = self._sample_classes(self.ref_probs.values, rng)

            sim_data = pd.DataFrame(
                {
                    self.id_col: self.ids,
                    self.strata_col: self.strata_classes,
                    "_map_class": sampled_map,
                    "_ref_class": sampled_ref,
                }
            )

            estimator = self.estimator_cls(
                sim_data,
                self.strata_col,
                "_map_class",
                "_ref_class",
                self.strata_population,
            )

            results["overall"][sim_idx] = estimator.overall_accuracy()[0]

            for k in self.all_classes:
                results["users"][k][sim_idx] = estimator.users_accuracy(k)[0]
                results["producers"][k][sim_idx] = estimator.producers_accuracy(k)[0]
                results["area"][k][sim_idx] = estimator.area(k, reference=True)[0]

            results["error_matrix"].append(estimator.error_matrix(proportions=True))

        self._results = results
        self._simulations_ran = True

    def _ensure_simulations(self):
        if not self._simulations_ran:
            self._run_simulations()

    def _summarize_distribution(self, distribution):
        finite = distribution[np.isfinite(distribution)]
        if finite.size == 0:
            return np.nan, (np.nan, np.nan)

        mean = float(np.mean(finite))
        lower, upper = np.percentile(finite, [2.5, 97.5])
        return mean, (float(lower), float(upper))

    def overall_accuracy(self):
        self._ensure_simulations()
        return self._summarize_distribution(self._results["overall"])

    def users_accuracy(self, k):
        self._ensure_simulations()
        return self._summarize_distribution(self._results["users"][k])

    def producers_accuracy(self, k):
        self._ensure_simulations()
        return self._summarize_distribution(self._results["producers"][k])

    def area(self, k, mapped=False, reference=True, correct=False):
        try:
            assert(sum([int(mapped), int(reference), int(correct)]) == 1)
        except AssertionError as error:
            msg = "exactly 1 of mapped, reference, and correct must be true"
            raise ValueError(msg) from error

        if mapped or correct:
            raise NotImplementedError

        self._ensure_simulations()
        return self._summarize_distribution(self._results["area"][k])

    def error_matrix(self, proportions=True):
        self._ensure_simulations()

        if not proportions:
            msg = "MCEM error_matrix only supports proportions=True"
            raise NotImplementedError(msg)

        matrix_sum = self._results["error_matrix"][0].copy()
        for matrix in self._results["error_matrix"][1:]:
            matrix_sum += matrix
        return matrix_sum / len(self._results["error_matrix"])

    def plot_distributions(self):
        self._ensure_simulations()

        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            msg = "matplotlib is required for plot_distributions"
            raise ImportError(msg) from error

        n_rows = len(self.all_classes) + 1
        fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3 * n_rows))
        axes = np.atleast_2d(axes)

        axes[0, 0].hist(self._results["overall"], bins=30)
        axes[0, 0].set_title("Overall Accuracy")
        axes[0, 1].axis("off")
        axes[0, 2].axis("off")

        for row_idx, k in enumerate(self.all_classes, start=1):
            axes[row_idx, 0].hist(self._results["users"][k], bins=30)
            axes[row_idx, 0].set_title(f"User's Accuracy: {k}")

            axes[row_idx, 1].hist(self._results["producers"][k], bins=30)
            axes[row_idx, 1].set_title(f"Producer's Accuracy: {k}")

            axes[row_idx, 2].hist(self._results["area"][k], bins=30)
            axes[row_idx, 2].set_title(f"Area: {k}")

        fig.tight_layout()
        return fig, axes