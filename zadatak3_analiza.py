import numpy as np
import pandas as pd
from poly_regression import PolynomialRegression

def evaluate_models():
    # Definisani podaci iz Tabele 1
    raw_data = {
        'Padavine_mm': [520, 610, 700, 820, 910, 1030, 1120, 1250, 1390, 1510],
        'Biomasa_tha': [3.1, 3.8, 4.0, 5.1, 5.3, 6.0, 6.2, 5.9, 5.1, 4.1]
    }
    dataframe = pd.DataFrame(raw_data)
    x_vals = dataframe['Padavine_mm'].values
    y_vals = dataframe['Biomasa_tha'].values

    # Testne tacke za proveru ekstrapolacije
    x_test = np.array([600, 900, 1300, 1600])

    print("=" * 70)
    print("ZADATAK 3: ANALIZA UTICAJA STEPENA POLINOMA M NA KVALITET MODELA")
    print("=" * 70)

    degrees_list = [1, 2, 3, 5, 9]

    for degree in degrees_list:
        estimator = PolynomialRegression(degree=degree, alpha=0.0)
        estimator.fit(x_vals, y_vals)
        
        train_mse = estimator.score(x_vals, y_vals)
        train_err = estimator.err_function(x_vals, y_vals)
        test_preds = estimator.predict(x_test)
        
        print(f"\n[Stepen M = {degree}]")
        print("Procenjeni koeficijenti w:", [round(val, 6) for val in estimator.coef_])
        print(f"Funkcija greske Err(w): {train_err:.6f}")
        print(f"Srednja kvadratna greska (MSE): {train_mse:.6f}")
        print("Predikcije na novim tackama:")
        for pt, pred in zip(x_test, test_preds):
            print(f"  X = {pt:4d} mm --> Predvidjena biomasa = {pred:.2f} t/ha")

if __name__ == "__main__":
    evaluate_models()
