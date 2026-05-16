from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance


def compute_shap_summary(model: object, features: pd.DataFrame, target: pd.Series | None = None) -> pd.DataFrame:
    try:
        import shap
    except ImportError:
        if target is None or not hasattr(model, "predict"):
            native = getattr(model, "feature_importances_", None)
            if native is not None:
                values = pd.Series(native, index=features.columns, name="mean_abs_shap")
                summary = values.sort_values(ascending=False).rename_axis("feature").reset_index()
                summary["importance_source"] = "native_feature_importance"
                return summary
            raise ImportError("Install the 'explainability' extra or provide target data for permutation importance.")
        result = permutation_importance(model, features, target, n_repeats=10, random_state=42)
        importance = pd.DataFrame(
            {
                "feature": features.columns,
                "mean_abs_shap": result.importances_mean,
                "importance_source": "permutation_importance",
            }
        )
        return importance.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    explainer = shap.Explainer(model)
    shap_values = explainer(features)
    values = pd.DataFrame(shap_values.values, index=features.index, columns=features.columns)
    summary = values.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").reset_index()
    summary.columns = ["feature", "mean_abs_shap"]
    summary["importance_source"] = "shap"
    return summary
