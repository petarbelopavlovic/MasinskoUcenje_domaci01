import numpy as np
import pandas as pd
from poly_regression import PolynomialRegression

def test_l2_regularization():
    raw_data = {
        'X': [520, 610, 700, 820, 910, 1030, 1120, 1250, 1390, 1510],
        'Y': [3.1, 3.8, 4.0, 5.1, 5.3, 6.0, 6.2, 5.9, 5.1, 4.1]
    }
    df = pd.DataFrame(raw_data)
    X_train, y_train = df['X'].values, df['Y'].values
    eval_points = np.array([600, 900, 1300, 1600])

    print("=" * 70)
    print("ZADATAK 4: PRIMENA L2 REGULARIZACIJE (RIDGE REGRESIJA)")
    print("=" * 70)

    for m in [3, 5, 9]:
        print(f"\n>>> Ispitivanje modela za M = {m} <<<")
        for alpha_param in [0.0, 1e-3, 1.0, 100.0]:
            ridge_fit = PolynomialRegression(degree=m, alpha=alpha_param)
            ridge_fit.fit(X_train, y_train)
            
            mse = ridge_fit.score(X_train, y_train)
            predictions = ridge_fit.predict(eval_points)
            w_norm = np.linalg.norm(ridge_fit.coef_)
            
            print(f"Alpha: {alpha_param:<7} | MSE: {mse:.6f} | ||w||: {w_norm:.3e} | Predikcije: {np.round(predictions, 2)}")

if __name__ == "__main__":
    test_l2_regularization()
