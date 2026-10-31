"""Kratak test skript za PolynomialRegression (Zadatak 2)."""
import numpy as np
import pandas as pd
from poly_regression import PolynomialRegression

df = pd.read_csv("data.csv")
X = df[["X"]].to_numpy()
y = df["Y"].to_numpy()

M = 2
model = PolynomialRegression(degree=M)
model.fit(X, y)

print(f"Stepen polinoma M = {M}")
print("Parametri w:", model.coef_)
y_hat = model.predict(X)
print("Predikcije na skupu za obuku:", np.round(y_hat, 3))
print("MSE (score):", model.score(X, y))

# Provera u odnosu na numpy.polyfit (referentno resenje)
ref = np.polyfit(X[:, 0], y, M)[::-1]
print("Maks. razlika u odnosu na np.polyfit:", np.max(np.abs(model.coef_ - ref)))

# Provera kompatibilnosti sa scikit-learn alatima
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
print("clone / get_params:", clone(model).get_params())
