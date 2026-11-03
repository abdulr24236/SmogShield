"""Shared data loading and splitting for Deliverable 2.

The processed feature table is created by
``data_scripts/09_build_features.py``. It is intentionally not committed to Git
because it is derived from external raw data.

This module enforces two rules that must be identical for every model:

1. The train/validation/test split is chronological, with a gap between
   periods. The target is tomorrow's value, so a random split would leak future
   information into training.
2. All imputers, encoders and scalers are fitted on the training split only.
   The ``*_scaled`` columns from Phase 2 are dropped because they were scaled
   on all rows and would leak validation/test statistics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the shared YAML configuration."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file is empty or invalid: {path}")
    return config


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_features(config: dict[str, Any]) -> pd.DataFrame:
    """Read and validate the processed feature table."""
    path = resolve_path(config["paths"]["features"])
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Restore it from the shared Drive or run the "
            "Phase 2 scripts (02 -> 03 -> 04 -> 08 -> 09) first."
        )

    df = pd.read_csv(path)
    date_col = config["data"]["date_column"]
    if date_col not in df.columns:
        raise KeyError(
            f"Feature table has no {date_col!r} column. Columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="raise")
    df = (
        df.sort_values(date_col)
        .drop_duplicates(subset=[date_col], keep="last")
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError(f"Feature table is empty: {path}")
    return df


@dataclass(frozen=True)
class ChronologicalSplit:
    """Three consecutive date ranges with a purge gap between them."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp
    gap_days: int
    date_column: str

    def summary(self) -> dict[str, Any]:
        date_col = self.date_column
        return {
            "train_rows": int(len(self.train)),
            "validation_rows": int(len(self.validation)),
            "test_rows": int(len(self.test)),
            "train_range": [
                str(self.train[date_col].min().date()),
                str(self.train[date_col].max().date()),
            ],
            "validation_range": [
                str(self.validation[date_col].min().date()),
                str(self.validation[date_col].max().date()),
            ],
            "test_range": [
                str(self.test[date_col].min().date()),
                str(self.test[date_col].max().date()),
            ],
            "gap_days": int(self.gap_days),
        }


def chronological_split(
    df: pd.DataFrame, config: dict[str, Any]
) -> ChronologicalSplit:
    """Split by date and purge the boundary days.

    With ``gap_days = 7`` and ``train_end = 2022-12-31``, the seven days
    2023-01-01 through 2023-01-07 are excluded and validation starts on
    2023-01-08. This prevents the next-day target at the end of one split from
    overlapping the start of the next split.
    """
    date_col = config["data"]["date_column"]
    split_cfg = config["split"]

    train_end = pd.Timestamp(split_cfg["train_end"])
    validation_end = pd.Timestamp(split_cfg["validation_end"])
    gap_days = int(split_cfg["gap_days"])
    test_end = split_cfg.get("test_end")
    test_end = pd.Timestamp(test_end) if test_end else df[date_col].max()

    if gap_days < 0:
        raise ValueError("split.gap_days must be non-negative.")
    if not (df[date_col].min() <= train_end < validation_end <= test_end):
        raise ValueError(
            "Invalid split dates: require data start <= train_end < "
            "validation_end <= test_end."
        )

    gap = pd.Timedelta(days=gap_days)
    train = df[df[date_col] <= train_end].copy()
    validation = df[
        (df[date_col] > train_end + gap) & (df[date_col] <= validation_end)
    ].copy()
    test = df[
        (df[date_col] > validation_end + gap) & (df[date_col] <= test_end)
    ].copy()

    empty = [
        name
        for name, frame in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
        if frame.empty
    ]
    if empty:
        raise ValueError(
            "Empty split(s): "
            + ", ".join(empty)
            + ". Adjust configs/base.yaml split dates or gap_days."
        )

    return ChronologicalSplit(
        train=train,
        validation=validation,
        test=test,
        train_end=train_end,
        validation_end=validation_end,
        test_end=test_end,
        gap_days=gap_days,
        date_column=date_col,
    )


def feature_columns(
    df: pd.DataFrame, config: dict[str, Any], target_column: str
) -> list[str]:
    """Return model input columns, excluding targets, date and duplicates."""
    data_cfg = config["data"]
    date_col = data_cfg["date_column"]

    drop = {date_col, target_column}
    drop.update(data_cfg.get("target_columns", []))
    drop.update(data_cfg.get("drop_columns", []))

    columns = [column for column in df.columns if column not in drop]
    suffixes = tuple(data_cfg.get("exclude_suffixes", []))
    if suffixes:
        columns = [column for column in columns if not column.endswith(suffixes)]

    if not columns:
        raise ValueError("No feature columns remain after applying the config.")

    # Hard leakage guard: do not rely only on config lists. A typo in
    # exclude_suffixes or target_columns must fail loudly instead of silently
    # feeding a future target or an all-row-scaled column to the model.
    leaky = [
        column
        for column in columns
        if column.startswith("target_") or column.endswith("_scaled")
    ]
    if leaky:
        raise ValueError(
            f"Leaky columns in feature set: {leaky}. "
            "Remove target_* and *_scaled columns from the model inputs."
        )

    categorical = [
        column
        for column in data_cfg.get("categorical_features", [])
        if column in columns
    ]
    numeric = [column for column in columns if column not in categorical]

    bad_numeric = [
        column
        for column in numeric
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if bad_numeric:
        raise TypeError(
            "Undeclared non-numeric feature columns: "
            f"{bad_numeric}. Add them to data.categorical_features or "
            "data.drop_columns in configs/base.yaml."
        )

    return columns


def make_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build the shared preprocessing pipeline.

    The returned transformer must be fitted on the training split only.
    """
    if not numeric_features and not categorical_features:
        raise ValueError("At least one numeric or categorical feature is required.")

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                                add_indicator=True,
                            ),
                        ),
                        ("scaler", RobustScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent",
                                add_indicator=True,
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def classification_labels(config: dict[str, Any]) -> list[str]:
    """Return the required, fixed class order used by classifiers/reports."""
    configured = config["data"].get("classification_labels")
    if not configured:
        raise ValueError(
            "Set data.classification_labels in ordinal order in "
            "configs/base.yaml; deriving labels from the data is not allowed."
        )

    labels = [str(label) for label in configured]
    if len(labels) < 2:
        raise ValueError("At least two classification labels are required.")
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate classification labels: {labels}")

    merge = config["data"].get("label_merge", {})
    if merge:
        if not isinstance(merge, dict):
            raise ValueError("data.label_merge must be a mapping of label -> label.")
        merged_values = {str(value) for value in merge.values()}
        missing = sorted(merged_values - set(labels))
        if missing:
            raise ValueError(
                f"label_merge produces labels absent from "
                f"classification_labels: {missing}"
            )
    return labels


def apply_label_merge(
    target: pd.Series, config: dict[str, Any]
) -> pd.Series:
    """Apply the configured class merge before encoding the target."""
    merge = config["data"].get("label_merge", {})
    if not merge:
        return target
    if not isinstance(merge, dict):
        raise ValueError("data.label_merge must be a mapping of label -> label.")
    mapping = {str(source): str(destination) for source, destination in merge.items()}
    return target.replace(mapping)


def encode_classification_target(
    target: pd.Series, labels: list[str]
) -> pd.Series:
    """Map AQI category strings to stable integer codes."""
    mapping = {label: index for index, label in enumerate(labels)}
    unknown = sorted(
        {str(value) for value in target.dropna().unique()} - set(mapping)
    )
    if unknown:
        raise ValueError(
            f"Target contains labels not present in config: {unknown}. "
            f"Configured labels: {labels}"
        )

    encoded = target.map(lambda value: mapping[str(value)] if pd.notna(value) else np.nan)
    if encoded.isna().any():
        raise ValueError("Classification target contains missing values.")
    return encoded.astype("int64")


def decode_classification_target(
    codes: np.ndarray | pd.Series, labels: list[str]
) -> np.ndarray:
    """Map integer predictions back to AQI category strings."""
    return np.asarray([labels[int(code)] for code in codes], dtype=object)


def class_counts(y: pd.Series, labels: list[str]) -> dict[str, int]:
    """Count each configured class in a split, including zero-count classes."""
    counts = y.value_counts().reindex(range(len(labels)), fill_value=0)
    return {
        label: int(counts.get(index, 0))
        for index, label in enumerate(labels)
    }


@dataclass(frozen=True)
class ModelData:
    """Prepared, leakage-safe model inputs for one task."""

    task: str
    target_name: str
    X_train_raw: pd.DataFrame
    X_validation_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    preprocessor: ColumnTransformer
    numeric_features: list[str]
    categorical_features: list[str]
    feature_names: list[str]
    labels: list[str] | None
    label_merge: dict[str, str]
    split: ChronologicalSplit

    def summary(self) -> dict[str, Any]:
        result = {
            "task": self.task,
            "target": self.target_name,
            "train_rows": int(len(self.X_train)),
            "validation_rows": int(len(self.X_validation)),
            "test_rows": int(len(self.X_test)),
            "n_features": int(len(self.feature_names)),
            "n_raw_features": int(
                len(self.numeric_features) + len(self.categorical_features)
            ),
            "split": self.split.summary(),
        }
        if self.label_merge:
            result["label_merge"] = self.label_merge
        if self.labels is not None:
            counts = {
                "train": class_counts(self.y_train, self.labels),
                "validation": class_counts(self.y_validation, self.labels),
                "test": class_counts(self.y_test, self.labels),
            }
            result["class_labels"] = self.labels
            result["class_counts"] = counts
            missing = {
                split_name: [
                    label for label, count in split_counts.items() if count == 0
                ]
                for split_name, split_counts in counts.items()
            }
            missing = {
                split_name: labels_missing
                for split_name, labels_missing in missing.items()
                if labels_missing
            }
            if missing:
                result["class_support_warnings"] = missing

        masks = self.scorable_rows
        result["scorable_rows"] = {
            name: int(mask.sum()) for name, mask in masks.items()
        }
        result["scorable_rows_dropped"] = {
            name: int((~mask).sum()) for name, mask in masks.items()
        }
        return result

    @property
    def scorable_rows(self) -> dict[str, pd.Series]:
        """Boolean masks for rows where today's ``pm25`` is finite.

        The masks are aligned with the raw frames and the target Series. Every
        model and baseline must score on the same mask, so a model cannot gain
        an advantage by imputing today's PM2.5 while persistence cannot use it.
        Do not use these masks to change training rows; they define evaluation
        rows only.
        """
        masks: dict[str, pd.Series] = {}
        for name, frame in (
            ("train", self.X_train_raw),
            ("validation", self.X_validation_raw),
            ("test", self.X_test_raw),
        ):
            if "pm25" not in frame.columns:
                raise KeyError(
                    "Cannot define scorable rows: current-day 'pm25' is not a "
                    "model feature."
                )
            values = pd.to_numeric(frame["pm25"], errors="coerce").to_numpy(
                dtype=float
            )
            masks[name] = pd.Series(
                np.isfinite(values),
                index=frame.index,
                name="scorable",
            )
        return masks

    def make_model_pipeline(self, estimator: Any) -> Pipeline:
        """Return an unfitted preprocessor + estimator pipeline for tuning.

        Use this with the raw frames (``X_train_raw``) and a time-series
        cross-validator. Do not cross-validate on ``X_train``: that matrix was
        already transformed with a preprocessor fitted on the whole training
        split, so each fold would see held-out scaling statistics.
        """
        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_preprocessor(
                        self.numeric_features, self.categorical_features
                    ),
                ),
                ("model", estimator),
            ]
        )


def prepare_model_data(
    config: dict[str, Any], task: str
) -> ModelData:
    """Load features, split by date, and fit preprocessing on train only.

    Parameters
    ----------
    config:
        Parsed ``configs/base.yaml``.
    task:
        ``"classification"`` or ``"regression"``.
    """
    if task == "classification":
        target_name = config["data"]["target_classification"]
    elif task == "regression":
        target_name = config["data"]["target_regression"]
    else:
        raise ValueError("task must be 'classification' or 'regression'.")

    df = load_features(config)
    if target_name not in df.columns:
        raise KeyError(
            f"Target {target_name!r} not found. Columns: {list(df.columns)}"
        )

    label_merge: dict[str, str] = {}
    if task == "classification":
        df[target_name] = df[target_name].replace(r"^\s*$", np.nan, regex=True)
        df[target_name] = apply_label_merge(df[target_name], config)
        merge = config["data"].get("label_merge", {})
        if merge:
            if not isinstance(merge, dict):
                raise ValueError(
                    "data.label_merge must be a mapping of label -> label."
                )
            label_merge = {
                str(source): str(destination)
                for source, destination in merge.items()
            }
    else:
        df[target_name] = pd.to_numeric(df[target_name], errors="coerce")

    df = df.dropna(subset=[target_name]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows remain after dropping missing {target_name!r}.")

    split = chronological_split(df, config)
    columns = feature_columns(df, config, target_name)
    categorical = [
        column
        for column in config["data"].get("categorical_features", [])
        if column in columns
    ]
    numeric = [column for column in columns if column not in categorical]

    preprocessor = make_preprocessor(numeric, categorical)

    X_train_raw = split.train[columns]
    X_validation_raw = split.validation[columns]
    X_test_raw = split.test[columns]

    X_train = preprocessor.fit_transform(X_train_raw)
    X_validation = preprocessor.transform(X_validation_raw)
    X_test = preprocessor.transform(X_test_raw)

    feature_names = list(preprocessor.get_feature_names_out())
    X_train = pd.DataFrame(
        X_train, columns=feature_names, index=X_train_raw.index
    )
    X_validation = pd.DataFrame(
        X_validation, columns=feature_names, index=X_validation_raw.index
    )
    X_test = pd.DataFrame(
        X_test, columns=feature_names, index=X_test_raw.index
    )

    labels: list[str] | None = None
    if task == "classification":
        labels = classification_labels(config)
        y_train = encode_classification_target(split.train[target_name], labels)
        y_validation = encode_classification_target(
            split.validation[target_name], labels
        )
        y_test = encode_classification_target(split.test[target_name], labels)

        for split_name, y_split in (
            ("train", y_train),
            ("validation", y_validation),
            ("test", y_test),
        ):
            missing = [
                label
                for index, label in enumerate(labels)
                if int((y_split == index).sum()) == 0
            ]
            if missing:
                warnings.warn(
                    f"{split_name} split has no rows for: {', '.join(missing)}. "
                    "Per-class metrics and one-vs-rest AUC for these classes "
                    "are undefined.",
                    UserWarning,
                    stacklevel=2,
                )
    else:
        y_train = split.train[target_name].astype(float)
        y_validation = split.validation[target_name].astype(float)
        y_test = split.test[target_name].astype(float)

    return ModelData(
        task=task,
        target_name=target_name,
        X_train_raw=X_train_raw,
        X_validation_raw=X_validation_raw,
        X_test_raw=X_test_raw,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        preprocessor=preprocessor,
        numeric_features=numeric,
        categorical_features=categorical,
        feature_names=feature_names,
        labels=labels,
        label_merge=label_merge,
        split=split,
    )
