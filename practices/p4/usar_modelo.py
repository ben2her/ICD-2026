"""Aplica el modelo guardado de Bank Marketing a un CSV.

Uso:
    python usar_modelo.py MODELO.joblib datos.csv [predicciones.csv]

`datos.csv` debe tener las columnas de bank-additional-full.csv (separador ';'); si trae `duration` o `y`, se ignoran.
Escribe un CSV con la columna `prediccion` (1 = yes, 0 = no) y `prediccion_etiqueta` (yes/no).

Por qué existen las dos definiciones de abajo: el modelo se entrenó en un notebook y su Pipeline usa una función
(`fe_bank`) y una clase (`WeightedBoost`) definidas ahí. Un archivo guardado con joblib.dump solo se puede cargar si
esas definiciones existen en el programa que lo abre, y deben estar en el módulo principal (__main__), como en este script.
Deben ser idénticas a las del notebook.
"""
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


def fe_bank(df):
    df = df.copy()
    df["no_prev_contact"] = (df["pdays"] == 999).astype(int)
    df["pdays"] = df["pdays"].where(df["pdays"] != 999, 0)
    return df


class WeightedBoost(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator=None, class_weight=None):
        self.estimator = estimator
        self.class_weight = class_weight

    def fit(self, X, y):
        self.le_ = LabelEncoder().fit(y)
        yi = self.le_.transform(y)
        sw = compute_sample_weight(self.class_weight, yi) if self.class_weight else None
        self.estimator_ = clone(self.estimator).fit(X, yi, sample_weight=sw)
        self.classes_ = self.le_.classes_
        return self

    def predict(self, X):
        return self.le_.inverse_transform(np.asarray(self.estimator_.predict(X)).astype(int).ravel())

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    modelo_path, datos_path = sys.argv[1], sys.argv[2]
    salida = sys.argv[3] if len(sys.argv) > 3 else "predicciones_bank.csv"

    modelo = joblib.load(modelo_path)
    df = pd.read_csv(datos_path, sep=";")
    X = df.drop(columns=[c for c in ("y", "duration") if c in df.columns])

    pred = modelo.predict(X)
    out = pd.DataFrame({"prediccion": pred, "prediccion_etiqueta": np.where(np.asarray(pred) == 1, "yes", "no")})
    out.to_csv(salida, index=False)
    print(f"{len(out)} filas -> {salida}")
    print(out["prediccion_etiqueta"].value_counts().to_string())
    if "y" in df.columns:
        from sklearn.metrics import f1_score, recall_score
        y = (df["y"] == "yes").astype(int)
        print(f"F1 macro contra la columna y del archivo: {f1_score(y, pred, average='macro'):.4f} "
              f"| sensibilidad de yes: {recall_score(y, pred):.3f}  (si el archivo es el de entrenamiento, es optimista)")


if __name__ == "__main__":
    main()
