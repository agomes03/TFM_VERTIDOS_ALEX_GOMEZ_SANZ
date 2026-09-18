"""
Fase 7b: XGBoost frente a Random Forest, mismo panel (viento ERA5) y mismo
esquema de validacion temporal en tres tramos (ajuste / calibracion / test),
para que la comparacion sea justa.

Ambos modelos se ajustan SIN reponderacion de clases y se calibran despues
por isotonica sobre el tramo intermedio -- misma leccion aprendida en la
primera version del Random Forest (reponderar descalibra las probabilidades).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                              precision_recall_curve, roc_curve)
from xgboost import XGBClassifier

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"

df = pd.read_csv(OUT_DIR / "panel_modelo_fase6_era5.csv")
df["has_event"] = (df["n_events"] > 0).astype(int)
df["anio"] = df["mes"].str.slice(0, 4).astype(int)
df["mes_num"] = df["mes"].str.slice(5, 7).astype(int)
df["mes_sin"] = np.sin(2 * np.pi * df["mes_num"] / 12)
df["mes_cos"] = np.cos(2 * np.pi * df["mes_num"] / 12)
df["anios_desde_2023"] = df["anio"] - 2023 + (df["mes_num"] - 1) / 12.0
df["log_ais_hours"] = np.log1p(df["ais_hours"])

df = df.sort_values(["cell_id", "mes"]).reset_index(drop=True)
df["lag1_has_event"] = df.groupby("cell_id")["has_event"].shift(1).fillna(0)
df["lag1_n_events"] = df.groupby("cell_id")["n_events"].shift(1).fillna(0)

FEATURES = ["wind_speed", "dist_costa_km", "log_ais_hours", "n_pases",
            "mes_sin", "mes_cos", "anios_desde_2023", "lag1_has_event", "lag1_n_events"]
df = df[df["n_pases"] > 0].dropna(subset=FEATURES + ["has_event"]).copy()

fit_ = df[df["mes"] < "2025-07"]
calib = df[(df["mes"] >= "2025-07") & (df["anio"] < 2026)]
test = df[df["anio"] >= 2026]
print(f"Ajuste: {len(fit_)} | Calibracion: {len(calib)} | Test: {len(test)}")
print(f"Tasas de positivos: {fit_['has_event'].mean():.4f} / {calib['has_event'].mean():.4f} / {test['has_event'].mean():.4f}")

X_fit, y_fit = fit_[FEATURES], fit_["has_event"]
X_cal, y_cal = calib[FEATURES], calib["has_event"]
X_test, y_test = test[FEATURES], test["has_event"]

baseline_rate = y_test.mean()
baseline_brier = baseline_rate * (1 - baseline_rate)

resultados = {}
curvas = {}
importancias = {}


def evaluar(nombre, modelo_base):
    modelo_base.fit(X_fit, y_fit)
    calibrado = CalibratedClassifierCV(FrozenEstimator(modelo_base), method="isotonic")
    calibrado.fit(X_cal, y_cal)

    proba = calibrado.predict_proba(X_test)[:, 1]
    proba_sin_cal = modelo_base.predict_proba(X_test)[:, 1]

    res = {
        "pr_auc": float(average_precision_score(y_test, proba)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "brier_sin_calibrar": float(brier_score_loss(y_test, proba_sin_cal)),
    }
    resultados[nombre] = res
    print(f"\n=== {nombre} ===")
    print(f"  PR-AUC:  {res['pr_auc']:.4f}   (baseline {baseline_rate:.4f})")
    print(f"  ROC-AUC: {res['roc_auc']:.4f}")
    print(f"  Brier:   {res['brier']:.4f}   (sin calibrar {res['brier_sin_calibrar']:.4f}, baseline {baseline_brier:.4f})")

    prec, rec, _ = precision_recall_curve(y_test, proba)
    curvas[nombre] = {"precision": prec[:-1], "recall": rec[:-1]}

    imp = pd.DataFrame({"variable": FEATURES, "importancia": modelo_base.feature_importances_})
    imp["importancia"] = imp["importancia"] / imp["importancia"].sum()
    importancias[nombre] = imp.sort_values("importancia", ascending=False)
    print(imp.sort_values("importancia", ascending=False).to_string(index=False))

    return proba


rf = RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=20,
                            n_jobs=-1, random_state=42)
proba_rf = evaluar("Random Forest", rf)

# Hiperparametros elegidos por busqueda con early stopping (tune_xgb.py):
# se entreno hasta 2024-12, se paro con un watch set 2025-01..2025-06 y se
# selecciono por PR-AUC sobre el tramo de calibracion (2025-07..2025-12).
# El test (2026) no se miro en ningun momento de la seleccion.
# Una primera configuracion sin ajustar (600 arboles, profundidad 6, sin early
# stopping) daba PR-AUC 0.304: sobreajustaba claramente.
xgb = XGBClassifier(
    n_estimators=262,          # best_iteration de la busqueda
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    eval_metric="aucpr",
    tree_method="hist",
)
proba_xgb = evaluar("XGBoost", xgb)

# --- Guardar resultados ---
comparativa = pd.DataFrame(resultados).T
comparativa.index.name = "modelo"
comparativa["baseline_pr_auc"] = baseline_rate
comparativa["baseline_brier"] = baseline_brier
comparativa.to_csv(OUT_DIR / "fase7b_comparativa_modelos.csv")
print("\n=== Comparativa ===")
print(comparativa.round(4).to_string())

for nombre, imp in importancias.items():
    slug = "rf" if "Forest" in nombre else "xgb"
    imp.to_csv(OUT_DIR / f"fase7b_importancia_{slug}.csv", index=False)

# Curvas PR, submuestreadas para las figuras
for nombre, c in curvas.items():
    slug = "rf" if "Forest" in nombre else "xgb"
    n = len(c["precision"])
    step = max(1, n // 300)
    pd.DataFrame({"precision": c["precision"][::step], "recall": c["recall"][::step]}).to_csv(
        OUT_DIR / f"fase7b_pr_curve_{slug}.csv", index=False)

# Predicciones del mejor modelo para el mapa de riesgo
mejor = max(resultados, key=lambda k: resultados[k]["pr_auc"])
print(f"\nMejor modelo por PR-AUC: {mejor}")
proba_mejor = proba_xgb if mejor == "XGBoost" else proba_rf
pred_out = test[["zona_estudio", "cell_id", "mes", "has_event"]].copy()
pred_out["proba_rf"] = proba_rf
pred_out["proba_xgb"] = proba_xgb
pred_out.to_csv(OUT_DIR / "fase7b_predicciones_2026_ambos.csv", index=False)

# Concordancia espacial entre ambos mapas de riesgo
riesgo = pred_out.groupby("cell_id")[["proba_rf", "proba_xgb"]].mean()
corr = riesgo["proba_rf"].corr(riesgo["proba_xgb"])
print(f"Correlacion entre los mapas de riesgo medio por celda (RF vs XGB): {corr:.3f}")

with open(OUT_DIR / "fase7b_resumen.txt", "w", encoding="utf-8") as f:
    f.write("=== Fase 7b: XGBoost vs Random Forest (viento ERA5) ===\n\n")
    f.write(f"Ajuste:      {len(fit_)} filas ({fit_['mes'].min()} - {fit_['mes'].max()})\n")
    f.write(f"Calibracion: {len(calib)} filas ({calib['mes'].min()} - {calib['mes'].max()})\n")
    f.write(f"Test:        {len(test)} filas ({test['mes'].min()} - {test['mes'].max()}), "
            f"{baseline_rate * 100:.2f}% positivos\n\n")
    f.write(comparativa.round(4).to_string())
    f.write(f"\n\nMejor modelo por PR-AUC: {mejor}\n")
    f.write(f"Correlacion entre mapas de riesgo por celda (RF vs XGB): {corr:.3f}\n\n")
    for nombre, imp in importancias.items():
        f.write(f"\nImportancia de variables - {nombre}:\n")
        f.write(imp.to_string(index=False))
        f.write("\n")

print(f"\nGuardado en {OUT_DIR}")
