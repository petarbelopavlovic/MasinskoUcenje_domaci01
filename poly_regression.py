import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

class PolynomialRegression(BaseEstimator, RegressorMixin):
    """
    Sopstveni regresioni model za polinomijalnu aproksimaciju sa L2 kaznenim clanom.
    Kompatibilan sa scikit-learn standardom.
    """
    def __init__(self, degree=1, alpha=0.0):
        self.degree = degree
        self.alpha = alpha
        self.coef_ = None

    def _build_design_matrix(self, input_vector):
        x_flat = np.asarray(input_vector).ravel()
        # Formiranje Vandermonde matrice reda N x (degree + 1)
        return np.vander(x_flat, N=self.degree + 1, increasing=True)

    def fit(self, X, y):
        design_phi = self._build_design_matrix(X)
        n_obs, n_params = design_phi.shape
        
        # Izracunavanje matrice sistema i uvodjenje regularizacije
        phi_t = design_phi.T
        matrix_a = phi_t @ design_phi
        
        # Matrix regularizacije L2 (bez uticaja na slobodni clan w0)
        reg_matrix = np.eye(n_params)
        reg_matrix[0, 0] = 0.0
        
        system_lhs = matrix_a + self.alpha * reg_matrix
        system_rhs = phi_t @ y
        
        # Analiticko nalazenje parametara w
        self.coef_ = np.linalg.lstsq(system_lhs, system_rhs, rcond=None)[0]
        return self

    def predict(self, X):
        mat_x = self._build_design_matrix(X)
        return np.matmul(mat_x, self.coef_)

    def score(self, X, y):
        preds = self.predict(X)
        errors = y - preds
        return float(np.mean(errors ** 2))

    def err_function(self, X, y):
        return 0.5 * self.score(X, y)
