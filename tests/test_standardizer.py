import pandas as pd
import pytest

from acc_assessment.standardizer import ProbStandardizer


def test_from_counts_raises_clear_error_for_missing_class_columns():
    standardizer = ProbStandardizer(class_names=["Forest", "Water", "Agriculture"])
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "Forest": [2, 1],
            "Water": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="Missing required class columns"):
        standardizer.from_counts(df)



def test_from_counts_duplicate_id_allowed_by_default():
    standardizer = ProbStandardizer(class_names=["Forest", "Water"])
    df = pd.DataFrame(
        {
            "id": [1, 1],
            "Forest": [2, 1],
            "Water": [1, 2],
        }
    )

    output = standardizer.from_counts(df)
    assert output.shape[0] == 2



def test_from_counts_duplicate_id_rejected_when_required():
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
        standardizer.from_counts(df)



def test_from_counts_missing_id_rejected_when_required():
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
        standardizer.from_counts(df)


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

    with pytest.raises(ValueError, match="must be in \[0, 1\]"):
        standardizer.from_confidence(df)
