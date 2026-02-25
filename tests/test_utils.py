import pytest
import numpy as np
import pandas as pd

from acc_assessment.utils import expand_error_matrix, load_integrated_probability_inputs


def test_expand_error_matrix_from_dict():
    error_matrices = {
        "A": pd.DataFrame(
            [[2, 1],
             [1, 2]],
            index=["F", "NF"],
            columns=["F", "NF"]
        ),
        "B": pd.DataFrame(
            [[4, 0],
             [3, 10]],
            index=["F", "NF"],
            columns=["F", "NF"],
        )
    }
    longform = expand_error_matrix(
        error_matrices,
        map_col="map",
        ref_col="ref",
        strata_col="strata",
    )

    assert longform.shape[0] == 23
    assert longform.shape[1] == 3
    assert longform.loc[longform["strata"] == "A"].shape[0] == 6
    assert longform.loc[longform["strata"] == "B"].shape[0] == 17
    assert longform.loc[longform["map"] == "F"].shape[0] == 7
    assert longform.loc[longform["map"] == "NF"].shape[0] == 16
    assert longform.loc[longform["ref"] == "F"].shape[0] == 10
    assert longform.loc[longform["ref"] == "NF"].shape[0] == 13


def test_load_integrated_probability_inputs_requires_separate_strata(tmp_path):
    map_df = pd.DataFrame(
        {
            "id": [1, 2],
            "strata": ["a", "b"],
            "A": [0.9, 0.2],
            "B": [0.1, 0.8],
        }
    )
    ref_df = pd.DataFrame({"id": [1, 2], "A": [0.8, 0.3], "B": [0.2, 0.7]})
    strata_sample_df = pd.DataFrame({"id": [1, 2], "strata": ["a", "b"]})
    strata_population_df = pd.DataFrame({"strata": ["a", "b"], "population": [5, 7]})

    map_path = tmp_path / "map.csv"
    ref_path = tmp_path / "ref.csv"
    sample_path = tmp_path / "sample_strata.csv"
    pop_path = tmp_path / "pop.csv"
    map_df.to_csv(map_path, index=False)
    ref_df.to_csv(ref_path, index=False)
    strata_sample_df.to_csv(sample_path, index=False)
    strata_population_df.to_csv(pop_path, index=False)

    with pytest.raises(ValueError, match="must not include 'strata'"):
        load_integrated_probability_inputs(
            map_file=map_path,
            ref_file=ref_path,
            strata_sample_file=sample_path,
            strata_population_file=pop_path,
        )


def test_load_integrated_probability_inputs_appends_strata_and_aligns_ref(tmp_path):
    map_df = pd.DataFrame({"id": [1, 2], "A": [0.9, 0.2], "B": [0.1, 0.8]})
    ref_df = pd.DataFrame({"id": [2, 1], "A": [0.3, 0.8], "B": [0.7, 0.2]})
    strata_sample_df = pd.DataFrame({"id": [1, 2], "strata": ["a", "b"]})
    strata_population_df = pd.DataFrame({"strata": ["a", "b"], "population": [5, 7]})

    map_path = tmp_path / "map.csv"
    ref_path = tmp_path / "ref.csv"
    sample_path = tmp_path / "sample_strata.csv"
    pop_path = tmp_path / "pop.csv"
    map_df.to_csv(map_path, index=False)
    ref_df.to_csv(ref_path, index=False)
    strata_sample_df.to_csv(sample_path, index=False)
    strata_population_df.to_csv(pop_path, index=False)

    map_out, ref_out, pop_df_out, pop_dict_out = load_integrated_probability_inputs(
        map_file=map_path,
        ref_file=ref_path,
        strata_sample_file=sample_path,
        strata_population_file=pop_path,
    )

    assert list(map_out.columns) == ["id", "A", "B", "strata"]
    assert list(ref_out.columns) == ["id", "A", "B", "strata"]
    assert map_out["id"].tolist() == [1, 2]
    assert ref_out["id"].tolist() == [1, 2]
    assert map_out["strata"].tolist() == ["a", "b"]
    assert ref_out["strata"].tolist() == ["a", "b"]
    assert pop_df_out.shape == (2, 2)
    assert pop_dict_out == {"a": 5, "b": 7}
