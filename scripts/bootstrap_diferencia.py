"""
Intervalo de confianza bootstrap para la DIFERENCIA de PR-AUC entre XGBoost y
Random Forest sobre el conjunto de test (2026). Sin esto, comparar 0.365 vs
0.361 no permite decir si un modelo es realmente mejor o es ruido muestral.

Se remuestrea el test con reemplazo (mismos indices para ambos modelos, para
que la comparacion sea pareada) y se recalcula la diferencia en cada replica.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"
rng = np.random.default_rng(42)
N_BOOT = 1000

pred = pd.read_csv(OUT_DIR / "fase7b_predicciones_2026_ambos.csv")
y = pred["has_event"].values
p_rf = pred["proba_rf"].values
p_xgb = pred["proba_xgb"].values
n = len(y)

obs_rf = average_precision_score(y, p_rf)
obs_xgb = average_precision_score(y, p_xgb)
obs_diff = obs_xgb - obs_rf
print(f"PR-AUC observado -> RF {obs_rf:.4f} | XGB {obs_xgb:.4f} | diferencia {obs_diff:+.4f}")

diffs = np.empty(N_BOOT)
rf_boot = np.empty(N_BOOT)
xgb_boot = np.empty(N_BOOT)
for i in range(N_BOOT):
    idx = rng.integers(0, n, n)
    yb = y[idx]
    if yb.sum() == 0:
        diffs[i] = np.nan
        continue
    a = average_precision_score(yb, p_rf[idx])
    b = average_precision_score(yb, p_xgb[idx])
    rf_boot[i], xgb_boot[i] = a, b
    diffs[i] = b - a
    if (i + 1) % 200 == 0:
        print(f"  {i + 1}/{N_BOOT} replicas")

diffs = diffs[~np.isnan(diffs)]
lo, hi = np.percentile(diffs, [2.5, 97.5])
p_xgb_mejor = float((diffs > 0).mean())

print(f"\nDiferencia PR-AUC (XGB - RF): {obs_diff:+.4f}")
print(f"IC 95% bootstrap: [{lo:+.4f}, {hi:+.4f}]")
print(f"Proporcion de replicas en que XGBoost supera a RF: {p_xgb_mejor:.3f}")
if lo <= 0 <= hi:
    veredicto = ("El intervalo de confianza incluye 0: la diferencia NO es distinguible "
                 "del ruido muestral. Ambos modelos son estadisticamente equivalentes.")
else:
    veredicto = "El intervalo de confianza excluye 0: la diferencia es consistente."
print(veredicto)

print(f"\nIC 95% PR-AUC RF:  [{np.percentile(rf_boot, 2.5):.4f}, {np.percentile(rf_boot, 97.5):.4f}]")
print(f"IC 95% PR-AUC XGB: [{np.percentile(xgb_boot, 2.5):.4f}, {np.percentile(xgb_boot, 97.5):.4f}]")

resumen = {
    "pr_auc_rf": obs_rf, "pr_auc_xgb": obs_xgb, "diferencia": obs_diff,
    "ic95_bajo": lo, "ic95_alto": hi, "prop_replicas_xgb_mejor": p_xgb_mejor,
    "rf_ic95_bajo": float(np.percentile(rf_boot, 2.5)), "rf_ic95_alto": float(np.percentile(rf_boot, 97.5)),
    "xgb_ic95_bajo": float(np.percentile(xgb_boot, 2.5)), "xgb_ic95_alto": float(np.percentile(xgb_boot, 97.5)),
    "n_boot": len(diffs),
}
pd.DataFrame([resumen]).to_csv(OUT_DIR / "fase7b_bootstrap_diferencia.csv", index=False)
pd.DataFrame({"diff": diffs}).to_csv(OUT_DIR / "fase7b_bootstrap_diffs.csv", index=False)

with open(OUT_DIR / "fase7b_bootstrap_resumen.txt", "w", encoding="utf-8") as f:
    f.write("=== Bootstrap de la diferencia de PR-AUC (XGBoost - Random Forest) ===\n\n")
    f.write(f"Replicas: {len(diffs)} (remuestreo pareado del test 2026, n={n})\n\n")
    f.write(f"PR-AUC Random Forest: {obs_rf:.4f}  IC95% [{np.percentile(rf_boot, 2.5):.4f}, {np.percentile(rf_boot, 97.5):.4f}]\n")
    f.write(f"PR-AUC XGBoost:       {obs_xgb:.4f}  IC95% [{np.percentile(xgb_boot, 2.5):.4f}, {np.percentile(xgb_boot, 97.5):.4f}]\n\n")
    f.write(f"Diferencia (XGB - RF): {obs_diff:+.4f}  IC95% [{lo:+.4f}, {hi:+.4f}]\n")
    f.write(f"Proporcion de replicas con XGBoost por delante: {p_xgb_mejor:.3f}\n\n")
    f.write(veredicto + "\n")

print(f"\nGuardado en {OUT_DIR}")
