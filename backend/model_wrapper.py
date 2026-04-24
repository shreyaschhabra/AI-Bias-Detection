import numpy as np
import pandas as pd


class ModelWrapper:
    def __init__(self, model, feature_names: list[str]):
        self.model = model
        self.feature_names = feature_names
        self._validate()

    def _validate(self):
        if not hasattr(self.model, "predict"):
            raise ValueError("Loaded object is not a valid sklearn estimator")
        if hasattr(self.model, "feature_names_in_"):
            missing = set(self.model.feature_names_in_) - set(self.feature_names)
            if missing:
                raise ValueError(f"CSV is missing model features: {missing}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.to_numpy())

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X.to_numpy())
        return None

    def feature_importances(self, feature_names: list[str]) -> dict[str, float]:
        """Return {feature: importance} for supported model types, normalized to sum=1."""
        if hasattr(self.model, "feature_importances_"):
            imps = np.array(self.model.feature_importances_, dtype=float)
            total = imps.sum() or 1.0
            return dict(zip(feature_names, imps / total))
        if hasattr(self.model, "coef_"):
            coefs = np.abs(self.model.coef_[0])
            total = coefs.sum() or 1.0
            return dict(zip(feature_names, coefs / total))
        return {}
