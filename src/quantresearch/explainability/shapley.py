from __future__ import annotations

import pandas as pd


def compute_shap_summary(model: object, features: pd.DataFrame) -> pd.DataFrame:
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install the 'explainability' extra to compute SHAP values.") from exc

    explainer = shap.Explainer(model)
    shap_values = explainer(features)
    values = pd.DataFrame(shap_values.values, index=features.index, columns=features.columns)
    return values.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").to_frame()
