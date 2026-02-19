import numpy as np
import pandas as pd

from pandas.testing import assert_frame_equal

from acc_assessment.gue import GUE
from acc_assessment.mcem import MCEM
from acc_assessment.stehman import Stehman


TOL = 1e-9


def _build_crisp_probability_tables(data, map_col, ref_col, strata_col, id_col="id"):
    classes = sorted(set(data[map_col].unique()) | set(data[ref_col].unique()))

    map_probs = pd.get_dummies(data[map_col]).reindex(columns=classes, fill_value=0)
    ref_probs = pd.get_dummies(data[ref_col]).reindex(columns=classes, fill_value=0)

    map_probs[strata_col] = data[strata_col].values
    ref_probs[strata_col] = data[strata_col].values

    ids = np.arange(data.shape[0])
    map_probs[id_col] = ids
    ref_probs[id_col] = ids
    return map_probs, ref_probs


def test_equivalence_constraint_stehman_gue_mcem():
    """Crisp probabilities must reproduce the crisp benchmark exactly."""
    data = pd.read_csv("./tests/stehman2014_table2.csv", skiprows=1)
    strata_population = {1: 40000, 2: 30000, 3: 20000, 4: 10000}
    map_probs, ref_probs = _build_crisp_probability_tables(
        data,
        map_col="Map class",
        ref_col="Reference class",
        strata_col="Stratum",
    )

    stehman = Stehman(
        data,
        strata_col="Stratum",
        map_col="Map class",
        ref_col="Reference class",
        strata_population=strata_population,
    )
    gue = GUE(
        map_data=map_probs,
        ref_data=ref_probs,
        strata_col="Stratum",
        id_col="id",
        strata_population=strata_population,
    )
    mcem = MCEM(
        map_data=map_probs,
        ref_data=ref_probs,
        strata_col="Stratum",
        id_col="id",
        strata_population=strata_population,
        n_simulations=100,
        random_state=0,
    )

    stehman_em = stehman.error_matrix(proportions=True)
    gue_em = gue.error_matrix(proportions=True)
    mcem_em = mcem.error_matrix(proportions=True)

    assert_frame_equal(stehman_em, gue_em, check_dtype=False, atol=TOL, rtol=0)
    assert_frame_equal(stehman_em, mcem_em, check_dtype=False, atol=TOL, rtol=0)

    stehman_oa, _ = stehman.overall_accuracy()
    gue_oa, _ = gue.overall_accuracy()
    mcem_oa, mcem_oa_ci = mcem.overall_accuracy()
    np.testing.assert_allclose([gue_oa, mcem_oa], [stehman_oa, stehman_oa], atol=TOL, rtol=0)

    np.testing.assert_allclose(
        [mcem_oa_ci[0], mcem_oa_ci[1]],
        [stehman_oa, stehman_oa],
        atol=TOL,
        rtol=0,
    )

    for class_name in ["A", "B", "C", "D"]:
        stehman_area, _ = stehman.area(class_name, reference=True)
        gue_area, _ = gue.area(class_name, reference=True)
        mcem_area, mcem_area_ci = mcem.area(class_name)

        np.testing.assert_allclose(
            [gue_area, mcem_area],
            [stehman_area, stehman_area],
            atol=TOL,
            rtol=0,
        )
        np.testing.assert_allclose(
            [mcem_area_ci[0], mcem_area_ci[1]],
            [stehman_area, stehman_area],
            atol=TOL,
            rtol=0,
        )
