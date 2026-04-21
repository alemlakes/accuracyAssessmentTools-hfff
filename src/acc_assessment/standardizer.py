import numpy as np
import pandas as pd


def verify_standard_structure(df, expected_classes, tolerance=0.001):
    """
    Check whether a dataframe matches the standard probabilistic structure.

    A valid structure has all expected class columns and each row of those
    columns sums to 1.0 within the provided tolerance.
    """
    missing = [col for col in expected_classes if col not in df.columns]
    if missing:
        return False

    class_block = df[expected_classes].apply(pd.to_numeric, errors="coerce")
    if class_block.isna().any(axis=None):
        return False

    row_sums = class_block.sum(axis=1)
    return bool(np.all(np.abs(row_sums - 1.0) <= tolerance))


class ProbStandardizer:
    """
    Convert non-standard confidence inputs into per-row class probabilities.
    """

    def __init__(
        self,
        class_names,
        id_col="id",
        strata_col="strata",
        require_unique_id=False,
    ):
        if len(class_names) == 0:
            raise ValueError("class_names must contain at least one class")
        self.class_names = list(class_names)
        self.id_col = id_col
        self.strata_col = strata_col
        self.require_unique_id = bool(require_unique_id)

    def _resolve_id_name(self, id_col=None):
        return self.id_col if id_col is None else id_col

    def _resolve_strata_name(self, strata_col=None):
        return self.strata_col if strata_col is None else strata_col

    def _class_block(self, df):
        missing = [col for col in self.class_names if col not in df.columns]
        if missing:
            missing_list = ", ".join(repr(col) for col in missing)
            raise ValueError(f"Missing required class columns: {missing_list}")
        return df[self.class_names]

    def _validate_unique_id(self, df, id_col=None):
        if not self.require_unique_id:
            return

        id_name = self._resolve_id_name(id_col=id_col)
        if id_name not in df.columns:
            raise ValueError(
                f"require_unique_id=True but id column {id_name!r} is missing"
            )

        if df[id_name].duplicated().any():
            raise ValueError(f"Duplicate ids found in column {id_name!r}")

    def _metadata_columns(self, df, id_col=None, strata_col=None):
        keep = []
        id_name = self._resolve_id_name(id_col=id_col)
        strata_name = self._resolve_strata_name(strata_col=strata_col)

        if strata_name in df.columns:
            keep.append(strata_name)
        if id_name in df.columns:
            keep.append(id_name)
        return keep

    def _copy_with_metadata(self, df, probs, id_col=None, strata_col=None):
        output = probs[self.class_names].copy()
        meta = self._metadata_columns(df, id_col=id_col, strata_col=strata_col)
        for col in meta:
            output[col] = df[col].values
        return output

    def _require_id_column(self, df, id_col=None, context="input"):
        id_name = self._resolve_id_name(id_col=id_col)
        if id_name not in df.columns:
            raise ValueError(f"{context} is missing required id column: {id_name!r}")
        return id_name

    def _normalize_rows(self, class_df):
        class_df = class_df.apply(pd.to_numeric, errors="coerce")
        if class_df.isna().any(axis=None):
            raise ValueError("Class columns contain missing or non-numeric values")

        row_sums = class_df.sum(axis=1)
        if np.any(row_sums <= 0):
            raise ValueError("Each row must have a positive total before normalization")

        return class_df.div(row_sums, axis=0)

    def from_likert(self, df, id_col=None, strata_col=None):
        self._validate_unique_id(df, id_col=id_col)
        class_df = self._class_block(df)
        probs = self._normalize_rows(class_df)
        return self._copy_with_metadata(df, probs, id_col=id_col, strata_col=strata_col)

    def from_votes(self, df, id_col=None, strata_col=None):
        self._validate_unique_id(df, id_col=id_col)
        class_df = self._class_block(df)
        probs = self._normalize_rows(class_df)
        return self._copy_with_metadata(df, probs, id_col=id_col, strata_col=strata_col)

    def from_multi_interpreter_vectors(self, df, id_col=None, strata_col=None):
        id_name = self._require_id_column(
            df,
            id_col=id_col,
            context="Multi-interpreter vector input",
        )
        strata_name = self._resolve_strata_name(strata_col=strata_col)

        class_df = self._class_block(df)
        probs = self._normalize_rows(class_df)

        grouped = probs.copy()
        grouped[id_name] = df[id_name].values
        grouped = grouped.groupby(id_name, sort=False)[self.class_names].mean().reset_index()

        if strata_name in df.columns:
            strata_df = df[[id_name, strata_name]].copy()
            n_unique = strata_df.groupby(id_name, sort=False)[strata_name].nunique(dropna=False)
            if (n_unique > 1).any():
                raise ValueError(
                    f"Each id must map to exactly one {strata_name!r} value in multi-interpreter vector input"
                )
            strata_df = strata_df.drop_duplicates(subset=[id_name])
            grouped = grouped.merge(strata_df, on=id_name, how="left", validate="one_to_one")

        output = grouped[self.class_names].copy()
        if strata_name in grouped.columns:
            output[strata_name] = grouped[strata_name].values
        output[id_name] = grouped[id_name].values
        return output

    def from_crisp(self, df, label_col="label", id_col=None, strata_col=None):
        self._validate_unique_id(df, id_col=id_col)

        if label_col not in df.columns:
            raise ValueError(f"Missing required column: {label_col}")

        labels = df[label_col]
        if labels.isna().any():
            raise ValueError(f"Column {label_col!r} contains missing values")

        class_set = set(self.class_names)
        unknown = sorted(set(labels) - class_set)
        if unknown:
            raise ValueError(
                "Crisp labels contain values not in class_names: "
                + ", ".join(repr(v) for v in unknown)
            )

        probs = pd.get_dummies(labels).reindex(columns=self.class_names, fill_value=0)
        output = probs.astype(float)
        return self._copy_with_metadata(df, output, id_col=id_col, strata_col=strata_col)

    def from_confidence(
        self,
        df,
        label_col="label",
        confidence_col="confidence",
        id_col=None,
        strata_col=None,
    ):
        self._validate_unique_id(df, id_col=id_col)

        if label_col not in df.columns:
            raise ValueError(f"Missing required column: {label_col}")
        if confidence_col not in df.columns:
            raise ValueError(f"Missing required column: {confidence_col}")

        labels = df[label_col]
        if labels.isna().any():
            raise ValueError(f"Column {label_col!r} contains missing values")

        confidences = pd.to_numeric(df[confidence_col], errors="coerce")
        if confidences.isna().any():
            raise ValueError(
                f"Column {confidence_col!r} contains missing or non-numeric values"
            )
        if ((confidences < 0) | (confidences > 1)).any():
            raise ValueError(f"Column {confidence_col!r} must be in [0, 1]")

        class_set = set(self.class_names)
        probs = np.zeros((len(df), len(self.class_names)), dtype=float)

        for idx, row in df.reset_index(drop=True).iterrows():
            label = row[label_col]
            if label not in class_set:
                raise ValueError(f"Label {label!r} is not in class_names")

            assigned = float(row[confidence_col])
            if len(self.class_names) == 1:
                probs[idx, 0] = 1.0
                continue

            remainder = 1.0 - assigned
            other_p = remainder / (len(self.class_names) - 1)
            probs[idx, :] = other_p
            probs[idx, self.class_names.index(label)] = assigned

        output = pd.DataFrame(probs, columns=self.class_names)
        return self._copy_with_metadata(df, output, id_col=id_col, strata_col=strata_col)

    def from_binary_confidence(
        self,
        df,
        label_col="label",
        is_confident_col="is_confident",
        high_p=0.99,
        low_p=0.40,
        id_col=None,
        strata_col=None,
    ):
        self._validate_unique_id(df, id_col=id_col)

        if label_col not in df.columns:
            raise ValueError(f"Missing required column: {label_col}")
        if is_confident_col not in df.columns:
            raise ValueError(f"Missing required column: {is_confident_col}")

        class_set = set(self.class_names)
        probs = np.zeros((len(df), len(self.class_names)), dtype=float)

        for idx, row in df.reset_index(drop=True).iterrows():
            label = row[label_col]
            is_confident = bool(row[is_confident_col])

            if label not in class_set:
                raise ValueError(f"Label {label!r} is not in class_names")

            assigned = float(high_p if is_confident else low_p)
            if assigned < 0 or assigned > 1:
                raise ValueError("high_p/low_p must be in [0, 1]")

            if len(self.class_names) == 1:
                probs[idx, 0] = 1.0
                continue

            remainder = 1.0 - assigned
            other_p = remainder / (len(self.class_names) - 1)
            probs[idx, :] = other_p
            probs[idx, self.class_names.index(label)] = assigned

        output = pd.DataFrame(probs, columns=self.class_names)
        return self._copy_with_metadata(df, output, id_col=id_col, strata_col=strata_col)

    def verify_standard_style(self, df, tolerance=0.001):
        return verify_standard_structure(df, self.class_names, tolerance=tolerance)
