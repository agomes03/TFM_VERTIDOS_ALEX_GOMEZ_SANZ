"""
Fase 7: modelo predictivo de probabilidad de deteccion por celda-mes.
Clasificacion (has_event) en vez de regresion de conteo: la pregunta 4 de la
hoja de ruta pide "probabilidad de deteccion", y el problema es de evento raro
(~1% positivos), por lo que PR-AUC y calibracion son las metricas relevantes,
no accuracy.

Validacion temporal: entrena en 2023-2025, valida en 2026 (no split aleatorio,
para que el modelo no "vea el futuro" de la misma celda en el mismo mes).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                              precision_recall_curve, classification_report)

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"

df = pd.read_csv(OUT_DIR / "panel_modelo_fase6_con_ais.csv")
print(f"Filas totales: {len(df)}")
df["has_event"] = (df["n_events"] > 0).astype(int)

df["anio"] = df["mes"].str.slice(0, 4).astype(int)
df["mes_num"] = df["mes"].str.slice(5, 7).astype(int)
df["mes_sin"] = np.sin(2 * np.pi * df["mes_num"] / 12)
df["mes_cos"] = np.cos(2 * np.pi * df["mes_num"] / 12)
df["anios_desde_2023"] = df["anio"] - 2023 + (df["mes_num"] - 1) / 12.0
df["log_ais_hours"] = np.log1p(df["ais_hours"])

# Historial previo por celda: si hubo deteccion el mes anterior (autocorrelacion espacial-temporal)
df = df.sort_values(["cell_id", "mes"]).reset_index(drop=True)
df["lag1_has_event"] = df.groupby("cell_id")["has_event"].shift(1)
df["lag1_n_events"] = df.groupby("cell_id")["n_events"].shift(1)
# Primer mes de cada celda no tiene lag -> se asume sin historial previo (0), no se descarta la fila
df["lag1_has_event"] = df["lag1_has_event"].fillna(0)
df["lag1_n_events"] = df["lag1_n_events"].fillna(0)

feature_cols = [
    "wind_speed", "dist_costa_km", "log_ais_hours", "n_pases",
    "mes_sin", "mes_cos", "anios_desde_2023",
    "lag1_has_event", "lag1_n_events",
]
df = df.dropna(subset=feature_cols + ["has_event"]).copy()
print(f"Filas tras dropna: {len(df)}")

# --- Split temporal en tres tramos: ajuste, calibracion, validacion final ---
# (entrenar y calibrar con class_weight="balanced" da buena discriminacion pero
# probabilidades mal calibradas -- se detecto al ver un Brier score PEOR que el
# de un modelo trivial que siempre predice la tasa base. Se corrige ajustando
# SIN reponderar clases y calibrando aparte con un tramo de tiempo posterior al
# ajuste pero anterior a la validacion, para no filtrar informacion del futuro).
fit_ = df[df["mes"] < "2025-07"]
calib = df[(df["mes"] >= "2025-07") & (df["anio"] < 2026)]
test = df[df["anio"] >= 2026]
print(f"Ajuste:      {len(fit_)} filas ({fit_['mes'].min()} - {fit_['mes'].max()}), "
      f"{fit_['has_event'].mean() * 100:.2f}% positivos")
print(f"Calibracion: {len(calib)} filas ({calib['mes'].min()} - {calib['mes'].max()}), "
      f"{calib['has_event'].mean() * 100:.2f}% positivos")
print(f"Test:        {len(test)} filas ({test['mes'].min()} - {test['mes'].max()}), "
      f"{test['has_event'].mean() * 100:.2f}% positivos")

X_fit, y_fit = fit_[feature_cols], fit_["has_event"]
X_calib, y_calib = calib[feature_cols], calib["has_event"]
X_test, y_test = test[feature_cols], test["has_event"]

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=20,
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_fit, y_fit)

rf_calibrado = CalibratedClassifierCV(FrozenEstimator(rf), method="isotonic")
rf_calibrado.fit(X_calib, y_calib)

proba_test = rf_calibrado.predict_proba(X_test)[:, 1]
proba_test_sin_calibrar = rf.predict_proba(X_test)[:, 1]

train = fit_  # para el bloque de importancias mas abajo, que usa el RF ya ajustado

pr_auc = average_precision_score(y_test, proba_test)
roc_auc = roc_auc_score(y_test, proba_test)
brier = brier_score_loss(y_test, proba_test)
brier_sin_calibrar = brier_score_loss(y_test, proba_test_sin_calibrar)
baseline_rate = y_test.mean()
baseline_brier = baseline_rate * (1 - baseline_rate)

print("\n=== Métricas de validación (2026, fuera de muestra) ===")
print(f"PR-AUC:  {pr_auc:.4f}  (baseline aleatorio = tasa base = {baseline_rate:.4f})")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Brier (calibrado):    {brier:.4f}")
print(f"Brier (sin calibrar): {brier_sin_calibrar:.4f}")
print(f"Brier baseline (predice siempre la tasa base): {baseline_brier:.4f}")

# Importancia de variables
importances = pd.DataFrame({
    "variable": feature_cols,
    "importancia": rf.feature_importances_,
}).sort_values("importancia", ascending=False)
print("\n=== Importancia de variables (Random Forest) ===")
print(importances.to_string(index=False))

# Curva precision-recall (para elegir un umbral operativo si se quisiera un mapa binario de riesgo)
precision, recall, thresholds = precision_recall_curve(y_test, proba_test)
pr_curve = pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1], "threshold": thresholds})
pr_curve.to_csv(OUT_DIR / "fase7_curva_precision_recall.csv", index=False)

importances.to_csv(OUT_DIR / "fase7_importancia_variables.csv", index=False)

# Guardar predicciones a nivel celda-mes para el mapa de riesgo (dashboard)
test_out = test[["zona_estudio", "cell_id", "mes", "has_event"]].copy()
test_out["proba_prevista"] = proba_test
test_out.to_csv(OUT_DIR / "fase7_predicciones_2026.csv", index=False)

with open(OUT_DIR / "fase7_resumen.txt", "w", encoding="utf-8") as f:
    f.write("=== Fase 7: modelo predictivo (Random Forest + calibracion isotonica) ===\n\n")
    f.write(f"Ajuste:      {len(fit_)} filas ({fit_['mes'].min()} - {fit_['mes'].max()}), "
            f"{fit_['has_event'].mean() * 100:.2f}% positivos\n")
    f.write(f"Calibracion: {len(calib)} filas ({calib['mes'].min()} - {calib['mes'].max()}), "
            f"{calib['has_event'].mean() * 100:.2f}% positivos\n")
    f.write(f"Test:        {len(test)} filas ({test['mes'].min()} - {test['mes'].max()}), "
            f"{test['has_event'].mean() * 100:.2f}% positivos\n\n")
    f.write(f"PR-AUC:  {pr_auc:.4f}  (baseline = {baseline_rate:.4f})\n")
    f.write(f"ROC-AUC: {roc_auc:.4f}\n")
    f.write(f"Brier (calibrado):    {brier:.4f}\n")
    f.write(f"Brier (sin calibrar): {brier_sin_calibrar:.4f}\n")
    f.write(f"Brier baseline:       {baseline_brier:.4f}\n\n")
    f.write("Importancia de variables:\n")
    f.write(importances.to_string(index=False))

print(f"\nGuardado en {OUT_DIR}: fase7_predicciones_2026.csv, fase7_importancia_variables.csv, "
      f"fase7_curva_precision_recall.csv, fase7_resumen.txt")
