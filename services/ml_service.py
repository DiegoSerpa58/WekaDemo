from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class AlgorithmSpec:
    id: str
    name: str
    description: str
    mode: str


ALGORITHMS: dict[str, AlgorithmSpec] = {
    "zeroR": AlgorithmSpec("zeroR", "ZeroR", "Predice siempre la clase mayoritaria.", "supervised"),
    "oneR": AlgorithmSpec("oneR", "OneR", "Regla simple basada en un solo atributo.", "supervised"),
    "j48": AlgorithmSpec("j48", "J48", "Árbol de decisión tipo C4.5.", "supervised"),
    "naiveBayes": AlgorithmSpec("naiveBayes", "Naive Bayes", "Clasificador probabilístico.", "supervised"),
    "random": AlgorithmSpec("random", "Random", "RandomForestClassifier.", "supervised"),
    "regresionMultiple": AlgorithmSpec(
        "regresionMultiple",
        "Regresión múltiple",
        "LinearRegression para objetivo numérico o LogisticRegression para clasificación.",
        "supervised",
    ),
    "rLogistica": AlgorithmSpec("rLogistica", "R.Logistica", "LogisticRegression.", "supervised"),
    "series": AlgorithmSpec("series", "Series", "MLPClassifier para patrones secuenciales.", "supervised"),
    "kmeans": AlgorithmSpec("kmeans", "Kmeans", "Clustering con KMeans.", "clustering"),
    "em": AlgorithmSpec("em", "EM", "Clustering con GaussianMixture.", "clustering"),
}


def get_algorithm_catalog() -> list[dict[str, str]]:
    return [
        {
            "id": algorithm.id,
            "name": algorithm.name,
            "description": algorithm.description,
            "mode": algorithm.mode,
        }
        for algorithm in ALGORITHMS.values()
    ]


def _build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError("Dataset does not contain valid feature columns.")

    return ColumnTransformer(transformers=transformers)


def _is_regression_target(target: pd.Series) -> bool:
    numeric_target = pd.to_numeric(target, errors="coerce")
    if numeric_target.isna().any():
        return False

    unique_count = numeric_target.nunique(dropna=True)
    return unique_count > 15


def _build_supervised_estimator(algorithm_id: str, is_regression: bool):
    if algorithm_id == "zeroR":
        return DummyClassifier(strategy="most_frequent")
    if algorithm_id == "oneR":
        return DecisionTreeClassifier(max_depth=1, random_state=1)
    if algorithm_id == "j48":
        return DecisionTreeClassifier(random_state=1)
    if algorithm_id == "naiveBayes":
        return GaussianNB()
    if algorithm_id == "random":
        return RandomForestClassifier(n_estimators=200, random_state=1)
    if algorithm_id == "regresionMultiple":
        return LinearRegression() if is_regression else LogisticRegression(max_iter=2000)
    if algorithm_id == "rLogistica":
        return LogisticRegression(max_iter=2000)
    if algorithm_id == "series":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=800, random_state=1)
    raise ValueError("Unsupported supervised algorithm.")


def _build_clustering_estimator(algorithm_id: str):
    if algorithm_id == "kmeans":
        return KMeans(n_clusters=3, random_state=1, n_init="auto")
    if algorithm_id == "em":
        return GaussianMixture(n_components=3, random_state=1)
    raise ValueError("Unsupported clustering algorithm.")


def _fit_predict_clustering(estimator, transformed_features: np.ndarray) -> np.ndarray:
    if isinstance(estimator, GaussianMixture):
        return estimator.fit_predict(transformed_features)
    return estimator.fit_predict(transformed_features)


def _run_clustering(dataset: pd.DataFrame, algorithm: AlgorithmSpec):
    if dataset.shape[1] < 1:
        raise ValueError("Dataset must contain at least one column.")

    features = dataset.iloc[:, :-1] if dataset.shape[1] > 1 else dataset.copy()
    preprocessor = _build_preprocessor(features)
    transformed_features = preprocessor.fit_transform(features)

    estimator = _build_clustering_estimator(algorithm.id)
    labels = _fit_predict_clustering(estimator, transformed_features)

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    silhouette = None
    if len(unique_labels) > 1 and len(labels) > len(unique_labels):
        silhouette = float(silhouette_score(transformed_features, labels))

    return {
        "taskType": "clustering",
        "algorithm": algorithm.name,
        "datasetName": "",
        "numInstances": int(len(dataset)),
        "numAttributes": int(len(dataset.columns)),
        "accuracy": 0.0,
        "kappa": 0.0,
        "meanAbsoluteError": 0.0,
        "confusionMatrix": [],
        "classNames": [],
        "precision": [],
        "recall": [],
        "fMeasure": [],
        "evaluationMethod": "Clustering",
        "clustering": {
            "clusterLabels": unique_labels.astype(str).tolist(),
            "clusterSizes": label_counts.astype(int).tolist(),
            "silhouette": silhouette,
        },
    }


def _minimum_class_count(target: pd.Series) -> int:
    return int(target.value_counts().min()) if not target.empty else 0


def _safe_cv_folds(target: pd.Series, requested_folds: int) -> int:
    minimum = _minimum_class_count(target)
    folds = min(requested_folds, minimum)
    if folds < 2:
        raise ValueError("Not enough data per class for cross-validation.")
    return folds


def _run_supervised(
    dataset: pd.DataFrame,
    algorithm: AlgorithmSpec,
    evaluation_method: str,
    folds: int,
    train_percent: float,
    seed: int,
):
    if dataset.shape[1] < 2:
        raise ValueError("Dataset must contain at least one feature and one target column.")

    features = dataset.iloc[:, :-1].copy()
    target = dataset.iloc[:, -1].copy()

    target_missing = target.isna()
    if target_missing.any():
        features = features.loc[~target_missing]
        target = target.loc[~target_missing]

    if target.empty:
        raise ValueError("Target column is empty after removing missing values.")

    is_regression = _is_regression_target(target)

    if algorithm.id in {"rLogistica", "series", "zeroR", "oneR", "j48", "naiveBayes", "random"} and is_regression:
        raise ValueError(f"{algorithm.name} requires a categorical target column.")

    estimator = _build_supervised_estimator(algorithm.id, is_regression)
    preprocessor = _build_preprocessor(features)
    model = Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])

    metrics: dict[str, Any]
    if evaluation_method == "percentagesplit":
        test_size = (100.0 - train_percent) / 100.0
        if not 0.01 <= test_size <= 0.99:
            raise ValueError("Train percentage must be between 1 and 99.")

        stratify_target = target if not is_regression and target.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_size,
            random_state=seed,
            stratify=stratify_target,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = _build_metrics(y_test, predictions, is_regression)
    else:
        if is_regression:
            cv_folds = max(2, min(folds, len(target)))
        else:
            cv_folds = _safe_cv_folds(target, folds)

        predictions = cross_val_predict(model, features, target, cv=cv_folds)
        metrics = _build_metrics(target, predictions, is_regression)

    response = {
        "taskType": "regression" if is_regression else "classification",
        "algorithm": algorithm.name,
        "datasetName": "",
        "numInstances": int(len(dataset)),
        "numAttributes": int(len(dataset.columns)),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "kappa": float(metrics.get("kappa", 0.0)),
        "meanAbsoluteError": float(metrics.get("mae", 0.0)),
        "confusionMatrix": metrics.get("confusionMatrix", []),
        "classNames": metrics.get("classNames", []),
        "precision": metrics.get("precision", []),
        "recall": metrics.get("recall", []),
        "fMeasure": metrics.get("fMeasure", []),
        "evaluationMethod": (
            f"crossvalidation ({folds} folds)" if evaluation_method == "crossvalidation" else f"Split {train_percent}%"
        ),
    }

    if is_regression:
        response["regression"] = {
            "mae": float(metrics["mae"]),
            "mse": float(metrics["mse"]),
            "r2": float(metrics["r2"]),
        }

    return response


def _build_metrics(true_values: pd.Series, predicted_values: np.ndarray, is_regression: bool) -> dict[str, Any]:
    if is_regression:
        true_numeric = pd.to_numeric(true_values, errors="coerce")
        pred_numeric = pd.to_numeric(pd.Series(predicted_values), errors="coerce")

        mask = ~(true_numeric.isna() | pred_numeric.isna())
        filtered_true = true_numeric.loc[mask]
        filtered_pred = pred_numeric.loc[mask]

        if filtered_true.empty:
            raise ValueError("Unable to compute regression metrics.")

        return {
            "accuracy": max(0.0, r2_score(filtered_true, filtered_pred) * 100.0),
            "kappa": 0.0,
            "mae": mean_absolute_error(filtered_true, filtered_pred),
            "mse": mean_squared_error(filtered_true, filtered_pred),
            "r2": r2_score(filtered_true, filtered_pred),
        }

    true_text = true_values.astype(str)
    pred_text = pd.Series(predicted_values).astype(str)
    class_names = sorted(set(true_text.tolist()) | set(pred_text.tolist()))

    if len(class_names) == 1:
        kappa = 0.0
    else:
        kappa = cohen_kappa_score(true_text, pred_text)

    confusion = confusion_matrix(true_text, pred_text, labels=class_names)
    precision, recall, f_measure, _ = precision_recall_fscore_support(
        true_text,
        pred_text,
        labels=class_names,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(true_text, pred_text) * 100.0,
        "kappa": float(kappa),
        "mae": 0.0,
        "confusionMatrix": confusion.tolist(),
        "classNames": class_names,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "fMeasure": f_measure.tolist(),
    }


def run_algorithm(
    dataset: pd.DataFrame,
    algorithm_id: str,
    evaluation_method: str,
    folds: int,
    train_percent: float,
    seed: int,
    dataset_name: str,
):
    if algorithm_id not in ALGORITHMS:
        raise ValueError("Unsupported algorithm.")

    algorithm = ALGORITHMS[algorithm_id]

    if algorithm.mode == "clustering":
        result = _run_clustering(dataset, algorithm)
    else:
        result = _run_supervised(dataset, algorithm, evaluation_method, folds, train_percent, seed)

    result["datasetName"] = dataset_name
    return result
