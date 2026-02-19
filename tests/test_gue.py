import numpy as np
import pandas as pd
import pytest

from acc_assessment.gue import GUE
from acc_assessment.stehman import Stehman


@pytest.fixture
def stehman_table():
    return pd.read_csv("./tests/stehman2014_table2.csv", skiprows=1)


@pytest.fixture
def strata_totals():
    return {1: 40000, 2: 30000, 3: 20000, 4: 10000}


@pytest.fixture
def crisp_prob_tables(stehman_table):
    class_names = sorted(
        set(stehman_table["Map class"].unique())
        | set(stehman_table["Reference class"].unique())
    )

    map_table = pd.get_dummies(stehman_table["Map class"])
    ref_table = pd.get_dummies(stehman_table["Reference class"])

    map_table = map_table.reindex(columns=class_names, fill_value=0)
    ref_table = ref_table.reindex(columns=class_names, fill_value=0)

    map_table["Stratum"] = stehman_table["Stratum"].values
    ref_table["Stratum"] = stehman_table["Stratum"].values

    map_table["id"] = np.arange(stehman_table.shape[0])
    ref_table["id"] = np.arange(stehman_table.shape[0])

    return map_table, ref_table


@pytest.fixture
def gue_assessment(crisp_prob_tables, strata_totals):
    map_table, ref_table = crisp_prob_tables
    return GUE(
        map_table,
        ref_table,
        id_col="id",
        strata_col="Stratum",
        strata_population=strata_totals,
    )


@pytest.fixture
def stehman_assessment(stehman_table, strata_totals):
    return Stehman(
        stehman_table,
        "Stratum",
        "Map class",
        "Reference class",
        strata_totals,
    )


def test_indicator_uses_probabilistic_joint_membership():
    map_table = pd.DataFrame(
        {
            "A": [0.7, 0.1],
            "B": [0.3, 0.9],
            "stratum": [1, 1],
            "id": [0, 1],
        }
    )
    ref_table = pd.DataFrame(
        {
            "A": [0.6, 0.2],
            "B": [0.4, 0.8],
            "stratum": [1, 1],
            "id": [0, 1],
        }
    )
    gue = GUE(
        map_table,
        ref_table,
        strata_col="stratum",
        id_col="id",
        strata_population={1: 2},
    )

    assert np.allclose(gue._indicator_func(map_val="A", ref_val="B"), [0.28, 0.08])
    assert np.allclose(gue._indicator_func(map_val="A"), [0.7, 0.1])
    assert np.allclose(gue._indicator_func(ref_val="B"), [0.4, 0.8])
    assert np.allclose(gue._indicator_func(), [0.54, 0.74])


def test_crisp_overall_accuracy_matches_stehman(gue_assessment, stehman_assessment):
    gue_value, gue_se = gue_assessment.overall_accuracy()
    stehman_value, stehman_se = stehman_assessment.overall_accuracy()
    assert np.isclose(gue_value, stehman_value)
    assert np.isclose(gue_se, stehman_se)


def test_crisp_error_matrix_matches_stehman(gue_assessment, stehman_assessment):
    gue_matrix = gue_assessment.error_matrix(proportions=True)
    stehman_matrix = stehman_assessment.error_matrix(proportions=True)
    assert np.allclose(gue_matrix.values, stehman_matrix.values)


@pytest.mark.parametrize("k", ["A", "B", "C", "D"])
def test_crisp_users_accuracy_matches_stehman(gue_assessment, stehman_assessment, k):
    gue_value, gue_se = gue_assessment.users_accuracy(k)
    stehman_value, stehman_se = stehman_assessment.users_accuracy(k)
    assert np.isclose(gue_value, stehman_value)
    assert np.isclose(gue_se, stehman_se)


@pytest.mark.parametrize("k", ["A", "B", "C", "D"])
def test_crisp_producers_accuracy_matches_stehman(
    gue_assessment,
    stehman_assessment,
    k,
):
    gue_value, gue_se = gue_assessment.producers_accuracy(k)
    stehman_value, stehman_se = stehman_assessment.producers_accuracy(k)
    assert np.isclose(gue_value, stehman_value)
    assert np.isclose(gue_se, stehman_se)


@pytest.mark.parametrize("k", ["A", "B", "C", "D"])
def test_crisp_area_estimates_match_stehman(gue_assessment, stehman_assessment, k):
    gue_pka, gue_pka_se = gue_assessment.PkA_estimate(k)
    stehman_pka, stehman_pka_se = stehman_assessment.PkA_estimate(k)
    assert np.isclose(gue_pka, stehman_pka)
    assert np.isclose(gue_pka_se, stehman_pka_se)

    gue_area, gue_area_se = gue_assessment.area(k, reference=True)
    stehman_area, stehman_area_se = stehman_assessment.area(k, reference=True)
    assert np.isclose(gue_area, stehman_area)
    assert np.isclose(gue_area_se, stehman_area_se)
