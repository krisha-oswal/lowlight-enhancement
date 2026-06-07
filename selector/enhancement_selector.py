"""
Enhancement Selector
====================
Novel Contribution: A lightweight model that predicts the best
enhancement method for a given image BEFORE detection runs.

Pipeline:
  1. Oracle Labeling: For each image, identify which enhancement
     method yielded the highest detection performance (per-image mAP/recall).
     This creates a supervised classification dataset.

  2. Feature Extraction: Use image characterization features
     (brightness, contrast, entropy, noise, histogram stats).

  3. Classifier Training: Train Random Forest, XGBoost, and MLP
     classifiers to predict the oracle label from image features.

  4. Evaluation:
     - Selector accuracy (how often does it predict the oracle method?)
     - Performance gap: selector vs oracle vs best fixed method
     - Feature importance analysis

Design rationale:
  Using image features as predictors makes physical sense:
  - Dark images (low brightness) → benefit from gamma/retinex
  - High noise → CLAHE may amplify noise; gamma safer
  - High entropy → already structured; baseline may be competitive
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Oracle Labeling
# ─────────────────────────────────────────────

def assign_oracle_labels(
    per_image_results: List[Dict],
    methods: List[str],
    metric: str = "recall",
    tie_break: str = "baseline",
) -> Dict[str, str]:
    """
    Assign oracle labels: for each image, which method gives best performance?

    Args:
        per_image_results: List of result dicts from evaluator
        methods: List of enhancement method names
        metric: 'recall' or 'precision' (per-image metric to maximize)
        tie_break: method to prefer when tied

    Returns:
        Dict mapping image_id → best_method_name
    """
    from collections import defaultdict

    # Group by image_id
    image_method_scores = defaultdict(dict)
    for result in per_image_results:
        img_id = result["image_id"]
        method = result["method"]
        score = result.get(metric, 0.0)
        image_method_scores[img_id][method] = score

    oracle_labels = {}
    for img_id, method_scores in image_method_scores.items():
        if not method_scores:
            oracle_labels[img_id] = tie_break
            continue

        best_score = max(method_scores.values())
        best_methods = [m for m, s in method_scores.items()
                        if abs(s - best_score) < 1e-5]

        # Tie-breaking: prefer tie_break method, then first alphabetically
        if tie_break in best_methods:
            oracle_labels[img_id] = tie_break
        else:
            oracle_labels[img_id] = sorted(best_methods)[0]

    return oracle_labels


def build_selector_dataset(
    per_image_results: List[Dict],
    oracle_labels: Dict[str, str],
    feature_prefix: str = "feat_",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and label vector y for selector training.

    Args:
        per_image_results: Per-image result list (contains features)
        oracle_labels: image_id → best_method
        feature_prefix: prefix for feature columns

    Returns:
        X: DataFrame of image features (one row per image)
        y: Series of oracle method labels
    """
    # Use only baseline results for features (original image characteristics)
    baseline_results = [r for r in per_image_results if r["method"] == "baseline"]

    rows = []
    labels = []
    image_ids = []

    for result in baseline_results:
        img_id = result["image_id"]
        if img_id not in oracle_labels:
            continue

        # Extract feature columns
        feat_dict = {}
        for key, val in result.items():
            if key.startswith(feature_prefix):
                feat_dict[key] = val
        # Also include quality fields if available
        for key in ["psnr", "ssim"]:
            if key in result:
                feat_dict[key] = result[key]

        if not feat_dict:
            continue

        rows.append(feat_dict)
        labels.append(oracle_labels[img_id])
        image_ids.append(img_id)

    if not rows:
        raise ValueError("No feature data found. Ensure baseline results contain features.")

    X = pd.DataFrame(rows, index=image_ids)
    y = pd.Series(labels, index=image_ids, name="oracle_method")

    # Fill any NaN
    X = X.fillna(0.0)

    logger.info(f"Selector dataset: {len(X)} samples, {len(X.columns)} features")
    logger.info(f"Label distribution:\n{y.value_counts()}")

    return X, y


# ─────────────────────────────────────────────
# Classifier Models
# ─────────────────────────────────────────────

def build_classifiers(config: dict) -> Dict[str, Any]:
    """Instantiate all configured classifiers."""
    sel_cfg = config.get("selector", {})
    classifiers = {}

    # Random Forest
    rf_cfg = sel_cfg.get("random_forest", {})
    classifiers["random_forest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 200),
            max_depth=rf_cfg.get("max_depth", 10),
            min_samples_split=rf_cfg.get("min_samples_split", 5),
            random_state=sel_cfg.get("random_state", 42),
            n_jobs=-1,
        ))
    ])

    # XGBoost (optional)
    try:
        import xgboost as xgb
        xgb_cfg = sel_cfg.get("xgboost", {})
        classifiers["xgboost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", xgb.XGBClassifier(
                n_estimators=xgb_cfg.get("n_estimators", 200),
                max_depth=xgb_cfg.get("max_depth", 6),
                learning_rate=xgb_cfg.get("learning_rate", 0.1),
                random_state=sel_cfg.get("random_state", 42),
                eval_metric="mlogloss",
                verbosity=0,
                use_label_encoder=False,
            ))
        ])
    except ImportError:
        logger.warning("XGBoost not installed. Skipping.")

    # MLP
    mlp_cfg = sel_cfg.get("mlp", {})
    classifiers["mlp"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=tuple(mlp_cfg.get("hidden_layer_sizes", [128, 64, 32])),
            max_iter=mlp_cfg.get("max_iter", 500),
            early_stopping=mlp_cfg.get("early_stopping", True),
            random_state=sel_cfg.get("random_state", 42),
        ))
    ])

    return classifiers


# ─────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────

class EnhancementSelector:
    """
    Trains and evaluates enhancement selector classifiers.

    Usage:
        selector = EnhancementSelector(config)
        selector.fit(X, y)
        results = selector.evaluate(X_test, y_test)
        best_method = selector.predict_best(image_features)
    """

    def __init__(self, config: dict):
        self.config = config
        self.sel_cfg = config.get("selector", {})
        self.random_state = self.sel_cfg.get("random_state", 42)
        self.test_size = self.sel_cfg.get("test_size", 0.2)
        self.cv_folds = self.sel_cfg.get("cross_val_folds", 5)

        self.classifiers = build_classifiers(config)
        self.label_encoder = LabelEncoder()

        self.results: Dict[str, Dict] = {}
        self.best_model_name: Optional[str] = None
        self.best_model = None

        # Train/test split (stored for evaluation)
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Train all classifiers with cross-validation.

        Args:
            X: Feature DataFrame
            y: Oracle label Series
        """
        # Encode labels
        y_enc = self.label_encoder.fit_transform(y)
        classes = self.label_encoder.classes_
        logger.info(f"Classes: {classes}")

        # Train/test split
        X_arr = X.values
        (self.X_train, self.X_test,
         self.y_train, self.y_test) = train_test_split(
            X_arr, y_enc,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_enc,
        )

        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        best_cv_score = -1.0

        for name, clf in self.classifiers.items():
            logger.info(f"Training {name}...")

            # Cross-validation
            cv_scores = cross_val_score(
                clf, self.X_train, self.y_train,
                cv=cv, scoring="accuracy", n_jobs=-1
            )

            # Final train on full training set
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)

            acc = accuracy_score(self.y_test, y_pred)
            report = classification_report(
                self.y_test, y_pred,
                target_names=classes,
                output_dict=True,
            )
            cm = confusion_matrix(self.y_test, y_pred)

            self.results[name] = {
                "cv_mean": float(np.mean(cv_scores)),
                "cv_std": float(np.std(cv_scores)),
                "test_accuracy": float(acc),
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
                "classes": list(classes),
            }

            logger.info(
                f"  {name}: CV={np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}, "
                f"Test Acc={acc:.3f}"
            )

            if np.mean(cv_scores) > best_cv_score:
                best_cv_score = np.mean(cv_scores)
                self.best_model_name = name
                self.best_model = clf

        logger.info(f"Best selector: {self.best_model_name} (CV={best_cv_score:.3f})")

    def predict_best(self, features: Dict[str, float]) -> str:
        """
        Predict best enhancement method for a new image.

        Args:
            features: Dict from extract_image_features()

        Returns:
            Predicted best method name
        """
        if self.best_model is None:
            return "baseline"

        feat_vec = np.array(list(features.values())).reshape(1, -1)
        pred_enc = self.best_model.predict(feat_vec)[0]
        return self.label_encoder.inverse_transform([pred_enc])[0]

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Extract feature importance from Random Forest (if available)."""
        clf_pipe = self.classifiers.get("random_forest")
        if clf_pipe is None:
            return None

        try:
            rf = clf_pipe.named_steps["clf"]
            importances = rf.feature_importances_
            return pd.Series(importances, name="importance")
        except Exception:
            return None

    def compute_performance_comparison(
        self,
        per_image_results: List[Dict],
        oracle_labels: Dict[str, str],
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = "recall",
    ) -> Dict[str, float]:
        """
        Compare:
          - Oracle performance (upper bound)
          - Selector performance
          - Best fixed method performance
          - Baseline performance

        Returns dict with mean metric scores for each strategy.
        """
        from collections import defaultdict

        # Per-image, per-method scores
        image_method_score = defaultdict(dict)
        for result in per_image_results:
            image_method_score[result["image_id"]][result["method"]] = \
                result.get(metric, 0.0)

        # Predict for test images
        test_image_ids = list(y.index[
            self.X_test is not None and
            slice(None)  # we'll recompute below
        ]) if hasattr(y, "index") else []

        # For each image in X (all), compute scores
        oracle_scores = []
        selector_scores = []
        baseline_scores = []

        for img_id in y.index:
            scores = image_method_score.get(img_id, {})
            if not scores:
                continue

            oracle_method = oracle_labels.get(img_id, "baseline")
            oracle_scores.append(scores.get(oracle_method, 0.0))
            baseline_scores.append(scores.get("baseline", 0.0))

            # Selector prediction (use best model)
            if img_id in X.index:
                feat_dict = dict(zip(X.columns, X.loc[img_id].values))
                pred_method = self.predict_best(feat_dict)
                selector_scores.append(scores.get(pred_method, 0.0))

        # Best fixed method
        method_mean_scores = {}
        for result in per_image_results:
            m = result["method"]
            if m not in method_mean_scores:
                method_mean_scores[m] = []
            method_mean_scores[m].append(result.get(metric, 0.0))
        best_fixed_method = max(
            method_mean_scores.keys(),
            key=lambda m: np.mean(method_mean_scores[m])
        )
        best_fixed_scores = method_mean_scores[best_fixed_method]

        return {
            "oracle_mean": float(np.mean(oracle_scores)) if oracle_scores else 0.0,
            "selector_mean": float(np.mean(selector_scores)) if selector_scores else 0.0,
            "best_fixed_method": best_fixed_method,
            "best_fixed_mean": float(np.mean(best_fixed_scores)) if best_fixed_scores else 0.0,
            "baseline_mean": float(np.mean(baseline_scores)) if baseline_scores else 0.0,
        }
