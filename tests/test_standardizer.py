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
