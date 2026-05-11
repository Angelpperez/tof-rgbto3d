# train_model.py
# Entrena clasificador XGBoost a nivel de nube de puntos y lo serializa.
#
# Uso:
#   python train_model.py --csv path/to/features_clean_con_posible_roca.csv
#
# Salida:
#   models/rock_classifier.joblib
#   models/rock_scaler.joblib
#
# Columnas esperadas en el CSV:
#   altura_norm, densidad, curvatura, normal_z, intensidad, posible_roca

import argparse
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

FEATURES = ["altura_norm", "densidad", "curvatura", "normal_z", "intensidad"]
LABEL    = "posible_roca"
OUT_DIR  = "models"


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en el CSV: {missing}")

    X = df[FEATURES].values.astype(np.float32)
    y = df[LABEL].values.astype(int)

    print(f"Dataset: {len(X)} muestras | rocas={y.sum()} | no-rocas={(1-y).sum()}")

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Ratio de clases para manejar desbalance
    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos = neg / pos if pos > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_sc, y, cv=cv, scoring="f1")
    print(f"CV F1 (5-fold): {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X_sc, y)
    y_pred = clf.predict(X_sc)
    print("\nReporte en entrenamiento completo:")
    print(classification_report(y, y_pred, target_names=["no-roca", "roca"]))

    # Importancia de features
    print("Importancia de features:")
    for feat, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(clf,    os.path.join(OUT_DIR, "rock_classifier.joblib"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "rock_scaler.joblib"))
    print(f"\nModelo guardado en {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Ruta al CSV con features_clean_con_posible_roca")
    args = parser.parse_args()
    main(args.csv)
