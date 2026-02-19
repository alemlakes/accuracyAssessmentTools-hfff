import numpy as np
import pandas as pd

from acc_assessment.mcem import MCEM
from acc_assessment.stehman import Stehman


def _crisp_probability_tables(data, map_col, ref_col, strata_col):
    classes = sorted(set(data[map_col].unique()) | set(data[ref_col].unique()))

    map_probs = pd.get_dummies(data[map_col]).reindex(columns=classes, fill_value=0)
    ref_probs = pd.get_dummies(data[ref_col]).reindex(columns=classes, fill_value=0)

    map_probs[strata_col] = data[strata_col].values
    ref_probs[strata_col] = data[strata_col].values

    map_probs["id"] = np.arange(data.shape[0])
    ref_probs["id"] = np.arange(data.shape[0])

    return map_probs, ref_probs


def test_mcem_crisp_matches_stehman():
    data = pd.read_csv("./tests/stehman2014_table2.csv", skiprows=1)
    strata_population = {1: 40000, 2: 30000, 3: 20000, 4: 10000}
    map_probs, ref_probs = _crisp_probability_tables(
        data,
        map_col="Map class",
        ref_col="Reference class",
        strata_col="Stratum",
    )

    mcem = MCEM(
        map_probs,
        ref_probs,
        strata_col="Stratum",
        id_col="id",
        strata_population=strata_population,
        n_simulations=100,
        random_state=0,
    )
    stehman = Stehman(
        data,
        "Stratum",
        "Map class",
        "Reference class",
        strata_population,
    )

    overall_mean, overall_ci = mcem.overall_accuracy()
    stehman_overall, _ = stehman.overall_accuracy()
    assert np.isclose(overall_mean, stehman_overall)
    assert np.isclose(overall_ci[0], stehman_overall)
    assert np.isclose(overall_ci[1], stehman_overall)

    for k in ["A", "B", "C", "D"]:
        ua_mean, ua_ci = mcem.users_accuracy(k)
        pa_mean, pa_ci = mcem.producers_accuracy(k)
        area_mean, area_ci = mcem.area(k)

        stehman_ua, _ = stehman.users_accuracy(k)
        stehman_pa, _ = stehman.producers_accuracy(k)
        stehman_area, _ = stehman.area(k, reference=True)

        assert np.isclose(ua_mean, stehman_ua)
        assert np.isclose(pa_mean, stehman_pa)
        assert np.isclose(area_mean, stehman_area)

        assert np.isclose(ua_ci[0], stehman_ua)
        assert np.isclose(ua_ci[1], stehman_ua)
        assert np.isclose(pa_ci[0], stehman_pa)
        assert np.isclose(pa_ci[1], stehman_pa)
        assert np.isclose(area_ci[0], stehman_area)
        assert np.isclose(area_ci[1], stehman_area)


def test_mcem_returns_mean_and_percentile_interval():
    map_data = pd.DataFrame(
        {
            "A": [0.8, 0.4, 0.3, 0.6],
            "B": [0.2, 0.6, 0.7, 0.4],
            "stratum": [1, 1, 2, 2],
            "id": [0, 1, 2, 3],
        }
    )
    ref_data = pd.DataFrame(
        {
            "A": [0.7, 0.5, 0.2, 0.4],
            "B": [0.3, 0.5, 0.8, 0.6],
            "stratum": [1, 1, 2, 2],
            "id": [0, 1, 2, 3],
        }
    )

    mcem = MCEM(
        map_data,
        ref_data,
        strata_col="stratum",
        id_col="id",
        strata_population={1: 500, 2: 500},
        n_simulations=200,
        random_state=42,
    )

    overall_mean, overall_ci = mcem.overall_accuracy()
    ua_mean, ua_ci = mcem.users_accuracy("A")
    pa_mean, pa_ci = mcem.producers_accuracy("A")
    area_mean, area_ci = mcem.area("A")

    assert isinstance(overall_mean, float)
    assert isinstance(overall_ci, tuple)
    assert len(overall_ci) == 2
    assert overall_ci[0] <= overall_mean <= overall_ci[1]

    for mean, ci in [(ua_mean, ua_ci), (pa_mean, pa_ci), (area_mean, area_ci)]:
        assert isinstance(mean, float)
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] <= mean <= ci[1]


def test_mcem_error_matrix_returns_simulation_mean():
    map_data = pd.DataFrame(
        {
            "A": [1.0, 0.0, 1.0, 0.0],
            "B": [0.0, 1.0, 0.0, 1.0],
            "stratum": [1, 1, 2, 2],
            "id": [0, 1, 2, 3],
        }
    )
    ref_data = map_data.copy()

    mcem = MCEM(
        map_data,
        ref_data,
        strata_col="stratum",
        id_col="id",
        strata_population={1: 500, 2: 500},
        n_simulations=20,
        random_state=12,
    )

    matrix = mcem.error_matrix()
    assert np.isclose(matrix.values.sum(), 1.0)
    assert np.isclose(np.trace(matrix.values), 1.0)