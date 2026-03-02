"""
ANN surrogate model for airfoil aerodynamic coefficients.

Uses scikit-learn MLPRegressor to approximate CL and CD as functions of
[alpha_deg, camber, thickness]. Separate regressors are trained for CL
and CD to avoid the precision degradation of multi-output models.

StandardScaler normalises both inputs and outputs before training so the
network never has to cope with magnitude differences between variables.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class ANNSurrogate:
    """
    Two-output ANN surrogate (CL, CD) built on sklearn MLPRegressor.

    Parameters
    ----------
    hidden_layers : tuple[int, ...]
        Number of neurons in each hidden layer.
    activation : str
        Activation function: "relu", "tanh", or "logistic".
    max_iter : int
        Maximum training iterations (epochs).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (64, 64, 32),
        activation: str = "relu",
        max_iter: int = 2000,
        random_state: int = 42,
    ) -> None:
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state

        # Scalers for inputs and each output
        self._scaler_X = StandardScaler()
        self._scaler_CL = StandardScaler()
        self._scaler_CD = StandardScaler()

        # Separate regressors for CL and CD
        self._model_CL: MLPRegressor | None = None
        self._model_CD: MLPRegressor | None = None

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _make_mlp(self) -> MLPRegressor:
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation=self.activation,
            solver="adam",
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50,
            tol=1e-5,
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def fit(
        self, X: np.ndarray, CL: np.ndarray, CD: np.ndarray
    ) -> dict[str, float]:
        """
        Fit the surrogate to DoE data.

        Parameters
        ----------
        X : (N, 3) array – [alpha_deg, camber, thickness]
        CL : (N,) array – lift coefficients
        CD : (N,) array – drag coefficients

        Returns
        -------
        dict with R² and MAE metrics for CL and CD on the training set.
        """
        from sklearn.metrics import mean_absolute_error, r2_score

        print(f"[surrogate] Training ANN {self.hidden_layers} on {len(X)} samples …")

        X_scaled = self._scaler_X.fit_transform(X)
        CL_scaled = self._scaler_CL.fit_transform(CL.reshape(-1, 1)).ravel()
        CD_scaled = self._scaler_CD.fit_transform(CD.reshape(-1, 1)).ravel()

        self._model_CL = self._make_mlp()
        self._model_CL.fit(X_scaled, CL_scaled)

        self._model_CD = self._make_mlp()
        self._model_CD.fit(X_scaled, CD_scaled)

        # Evaluate on training data
        CL_pred, CD_pred = self.predict(X)
        metrics = {
            "r2_CL":  float(r2_score(CL, CL_pred)),
            "r2_CD":  float(r2_score(CD, CD_pred)),
            "mae_CL": float(mean_absolute_error(CL, CL_pred)),
            "mae_CD": float(mean_absolute_error(CD, CD_pred)),
        }
        print(
            f"[surrogate] R²: CL={metrics['r2_CL']:.4f}, CD={metrics['r2_CD']:.4f}"
            f"  MAE: CL={metrics['mae_CL']:.4f}, CD={metrics['mae_CD']:.6f}"
        )
        return metrics

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict CL and CD for a batch of design points.

        Parameters
        ----------
        X : (N, 3) array – [alpha_deg, camber, thickness]

        Returns
        -------
        CL_pred : (N,) array
        CD_pred : (N,) array
        """
        if self._model_CL is None or self._model_CD is None:
            raise RuntimeError("ANNSurrogate.fit() must be called before predict().")

        X_scaled = self._scaler_X.transform(X)
        CL_scaled = self._model_CL.predict(X_scaled)
        CD_scaled = self._model_CD.predict(X_scaled)

        CL_pred = self._scaler_CL.inverse_transform(CL_scaled.reshape(-1, 1)).ravel()
        CD_pred = self._scaler_CD.inverse_transform(CD_scaled.reshape(-1, 1)).ravel()
        return CL_pred, CD_pred

    def evaluate(
        self, alpha_deg: float, camber: float, thickness: float
    ) -> dict[str, Any]:
        """
        Evaluate a single design point and return a dict compatible with the
        mock_evaluator interface so ANNSurrogate can be used as a drop-in
        replacement for ``_evaluate_fn`` in AirfoilProblem.

        Parameters
        ----------
        alpha_deg : float
        camber : float
        thickness : float

        Returns
        -------
        dict with keys: CL, CD, CL_CD, converged
        """
        X = np.array([[alpha_deg, camber, thickness]])
        CL_pred, CD_pred = self.predict(X)
        CL = float(CL_pred[0])
        CD = float(max(CD_pred[0], 1e-9))  # prevent division-by-zero
        return {
            "CL": CL,
            "CD": CD,
            "CL_CD": CL / CD,
            "converged": True,
        }

    def save(self, path: str) -> None:
        """Persist the trained surrogate to a pickle file."""
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[surrogate] Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "ANNSurrogate":
        """Load a previously saved surrogate from a pickle file."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[surrogate] Model loaded ← {path}")
        return obj
