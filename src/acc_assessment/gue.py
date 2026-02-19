import numpy as np
import pandas as pd

from acc_assessment.stehman import Stehman
from acc_assessment.utils import users_accuracy_error, producers_accuracy_error


class GUE(Stehman):
    """
    Generalized Unbiased Estimator (GUE): probabilistic extension of Stehman
    design-based accuracy assessment where point membership in each cell of the
    error matrix is represented by expected joint probabilities.
    """

    def __init__(
        self,
        map_data,
        ref_data,
        strata_col,
        id_col,
        strata_population,
    ):
        """
        map_data/ref_data: DataFrames containing class probability vectors,
            along with `strata_col` and `id_col`.
        strata_col: Name of the stratum label column in both dataframes.
        id_col: Name of unique point id column in both dataframes.
        strata_population: Mapping of stratum -> total stratum population.
        """
        assert np.all(map_data.shape == ref_data.shape)
        assert (map_data[id_col] == ref_data[id_col]).all()
        assert (map_data.columns == ref_data.columns).all()

        self.strata_classes = map_data[strata_col].values

        self.class_names = [
            col for col in map_data.columns if col not in [strata_col, id_col]
        ]

        self.map_probs = self._normalize_probabilities(
            map_data[self.class_names].astype(float).reset_index(drop=True)
        )
        self.ref_probs = self._normalize_probabilities(
            ref_data[self.class_names].astype(float).reset_index(drop=True)
        )

        self.map_classes = np.array(self.class_names)
        self.ref_classes = np.array(self.class_names)
        self.all_classes = np.array(self.class_names)

        self.strata_population = {
            k: v for k, v in iter(strata_population.items())
            if k in np.unique(self.strata_classes)
        }
        self.N = np.sum(list(self.strata_population.values()))

    @staticmethod
    def _normalize_probabilities(probs_df):
        probs = probs_df.values.astype(float)
        row_sums = probs.sum(axis=1)
        if np.any(row_sums <= 0):
            msg = "Each row of class probabilities must have a positive sum"
            raise ValueError(msg)
        normalized = probs / row_sums[:, None]
        return pd.DataFrame(normalized, columns=probs_df.columns)

    def _indicator_func(self, map_val=None, ref_val=None):
        """
        Expected indicator function under probabilistic class membership.
        """
        if map_val is not None and ref_val is not None:
            return (
                self.map_probs[map_val].values * self.ref_probs[ref_val].values
            ).astype(float)

        if map_val is not None:
            return self.map_probs[map_val].values.astype(float)

        if ref_val is not None:
            return self.ref_probs[ref_val].values.astype(float)

        return np.sum(
            self.map_probs.values * self.ref_probs.values,
            axis=1,
        ).astype(float)

    def _proportions_error_matrix(self):
        matrix = np.zeros((len(self.all_classes), len(self.all_classes)))
        for i, map_class in enumerate(self.all_classes):
            for j, ref_class in enumerate(self.all_classes):
                matrix[i, j] = self.Pij_estimate(map_class, ref_class)[0]
        return pd.DataFrame(matrix, columns=self.all_classes, index=self.all_classes)

    def _counts_error_matrix(self):
        matrix = np.zeros((len(self.all_classes), len(self.all_classes)))
        for i, map_class in enumerate(self.all_classes):
            for j, ref_class in enumerate(self.all_classes):
                matrix[i, j] = np.sum(
                    self.map_probs[map_class].values * self.ref_probs[ref_class].values
                )
        return pd.DataFrame(matrix, columns=self.all_classes, index=self.all_classes)

    def error_matrix(self, proportions=True):
        if proportions:
            return self._proportions_error_matrix()
        return self._counts_error_matrix()

    def users_accuracy(self, k):
        """
        Probabilistic user's accuracy for class k.

        Uses expected joint membership P(map=k, ref=k) in the numerator and
        expected mapped membership P(map=k) in the denominator.
        """
        y_u = self._indicator_func(map_val=k, ref_val=k)
        x_u = self._indicator_func(map_val=k)
        if np.isclose(np.sum(x_u), 0.0, atol=1e-12):
            users_accuracy_error(k)
        return self._design_consistent_estimator(y_u, x_u)

    def producers_accuracy(self, k):
        """
        Probabilistic producer's accuracy for class k.

        Uses expected joint membership P(map=k, ref=k) in the numerator and
        expected reference membership P(ref=k) in the denominator.
        """
        y_u = self._indicator_func(map_val=k, ref_val=k)
        x_u = self._indicator_func(ref_val=k)
        if np.isclose(np.sum(x_u), 0.0, atol=1e-12):
            producers_accuracy_error(k)
        return self._design_consistent_estimator(y_u, x_u)

    def commission_error_rate(self, k):
        acc, se = self.users_accuracy(k)
        return 1 - acc, se

    def omission_error_rate(self, k):
        acc, se = self.producers_accuracy(k)
        return 1 - acc, se
