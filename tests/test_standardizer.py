import pandas as pd
import pytest

from acc_assessment.standardizer import ProbStandardizer


def test_from_votes_matches_row_normalized_counts():
    standardizer = ProbStandardizer(class_names=["Forest", "Water", "Agriculture"])
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "Forest": [2, 1],
            "Water": [1, 2],
            "Agriculture": [1, 1],
        }
    )

    output = standardizer.from_votes(df)
    expected = pd.DataFrame(
        {
            "Forest": [0.5, 0.25],
            "Water": [0.25, 0.5],
            "Agriculture": [0.25, 0.25],
            "id": [1, 2],
        }
    )

    pd.testing.assert_frame_equal(output.reset_index(drop=True), expected)


def test_from_votes_raises_clear_error_for_missing_class_columns():
    standardizer = ProbStandardizer(class_names=["Forest", "Water", "Agriculture"])
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "Forest": [2, 1],
            "Water": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="Missing required class columns"):
        standardizer.from_votes(df)



def test_from_votes_duplicate_id_allowed_by_default():
    standardizer = ProbStandardizer(class_names=["Forest", "Water"])
    df = pd.DataFrame(
        {
            "id": [1, 1],
            "Forest": [2, 1],
            "Water": [1, 2],
        }
    )

    output = standardizer.from_votes(df)
    assert output.shape[0] == 2



def test_from_votes_duplicate_id_rejected_when_required():
    standardizer = ProbStandardizer(
        class_names=["Forest", "Water"],
        require_unique_id=True,
    )
    df = pd.DataFrame(
        {
            "id": [1, 1],
            "Forest": [2, 1],
            "Water": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="Duplicate ids found"):
        standardizer.from_votes(df)



def test_from_votes_missing_id_rejected_when_required():
    standardizer = ProbStandardizer(
        class_names=["Forest", "Water"],
        require_unique_id=True,
    )
    df = pd.DataFrame(
        {
            "Forest": [2, 1],
            "Water": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="id column"):
        standardizer.from_votes(df)


def test_from_multi_interpreter_vectors_averages_normalized_vectors_by_id():
    standardizer = ProbStandardizer(class_names=["Growth", "Loss", "Stable"])
    df = pd.DataFrame(
        {
            "id": [10, 10, 11],
            "strata": ["a", "a", "b"],
            "Growth": [0.8, 0.2, 0.1],
            "Loss": [0.1, 0.5, 0.2],
            "Stable": [0.1, 0.3, 0.7],
        }
    )

    output = standardizer.from_multi_interpreter_vectors(df)
    expected = pd.DataFrame(
        {
            "Growth": [0.5, 0.1],
            "Loss": [0.3, 0.2],
            "Stable": [0.2, 0.7],
            "strata": ["a", "b"],
            "id": [10, 11],
        }
    )

    pd.testing.assert_frame_equal(output.reset_index(drop=True), expected)


def test_from_multi_interpreter_vectors_rejects_inconsistent_strata_within_id():
    standardizer = ProbStandardizer(class_names=["Forest", "Water"])
    df = pd.DataFrame(
        {
            "id": [1, 1],
            "strata": ["a", "b"],
            "Forest": [0.8, 0.2],
            "Water": [0.2, 0.8],
        }
    )

    with pytest.raises(ValueError, match="exactly one 'strata' value"):
        standardizer.from_multi_interpreter_vectors(df)


def test_from_multi_interpreter_vectors_requires_id_column():
    standardizer = ProbStandardizer(class_names=["Forest", "Water"])
    df = pd.DataFrame(
        {
            "Forest": [0.8, 0.2],
            "Water": [0.2, 0.8],
        }
    )

    with pytest.raises(ValueError, match="required id column"):
        standardizer.from_multi_interpreter_vectors(df)


def test_from_crisp_converts_to_one_hot_probabilities():
    standardizer = ProbStandardizer(class_names=["Forest", "Water", "Agriculture"])
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "strata": ["a", "a", "f"],
            "label": ["Forest", "Water", "Agriculture"],
        }
    )

    output = standardizer.from_crisp(df, label_col="label")
    expected = pd.DataFrame(
        {
            "Forest": [1.0, 0.0, 0.0],
            "Water": [0.0, 1.0, 0.0],
            "Agriculture": [0.0, 0.0, 1.0],
            "strata": ["a", "a", "f"],
            "id": [1, 2, 3],
        }
    )

    pd.testing.assert_frame_equal(output.reset_index(drop=True), expected)


def test_from_crisp_rejects_unknown_label_values():
    standardizer = ProbStandardizer(class_names=["Forest", "Water"])
    df = pd.DataFrame({"label": ["Forest", "Urban"]})

    with pytest.raises(ValueError, match="not in class_names"):
        standardizer.from_crisp(df, label_col="label")


def test_from_confidence_uses_row_confidence_values():
    standardizer = ProbStandardizer(class_names=["Forest", "Water", "Agriculture"])
    df = pd.DataFrame(
        {
            "id": [34, 35],
            "strata": ["a", "f"],
            "label": ["Water", "Forest"],
            "confidence": [0.6, 0.9],
        }
    )

    output = standardizer.from_confidence(
        df,
        label_col="label",
        confidence_col="confidence",
    )

    expected = pd.DataFrame(
        {
            "Forest": [0.2, 0.9],
            "Water": [0.6, 0.05],
            "Agriculture": [0.2, 0.05],
            "strata": ["a", "f"],
            "id": [34, 35],
        }
    )

    pd.testing.assert_frame_equal(output.reset_index(drop=True), expected)


def test_from_confidence_rejects_values_outside_unit_interval():
    standardizer = ProbStandardizer(class_names=["Forest", "Water"])
    df = pd.DataFrame(
        {
            "label": ["Forest", "Water"],
            "confidence": [0.7, 1.2],
        }
    )

    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        standardizer.from_confidence(df)
