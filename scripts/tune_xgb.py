"""
Busqueda de hiperparametros para XGBoost antes de declararlo peor que
Random Forest. Se reserva la ultima parte del tramo de AJUSTE como conjunto
de watch para early stopping, dejando el tramo de calibracion intacto (solo
para calibrar) y el test (2026) sin tocar.
"""
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
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

# Ajuste propiamente dicho hasta 2024-12; watch = 2025-01..2025-06 (early stopping)
train = df[df["mes"] < "2025-01"]
watch = df[(df["mes"] >= "2025-01") & (df["mes"] < "2025-07")]
calib = df[(df["mes"] >= "2025-07") & (df["anio"] < 2026)]

print(f"train {len(train)} | watch {len(watch)} | calib {len(calib)}")

X_tr, y_tr = train[FEATURES], train["has_event"]
X_w, y_w = watch[FEATURES], watch["has_event"]
X_c, y_c = calib[FEATURES], calib["has_event"]

grid = {
    "max_depth": [3, 4, 6],
    "learning_rate": [0.03, 0.1],
    "min_child_weight": [5, 30],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_lambda": [1.0, 10.0],
}

keys = list(grid)
resultados = []
for combo in product(*(grid[k] for k in keys)):
    params = dict(zip(keys, combo))
    model = XGBClassifier(
        n_estimators=2000, n_jobs=-1, random_state=42,
        eval_metric="aucpr", tree_method="hist",
        early_stopping_rounds=50, **params,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_w, y_w)], verbose=False)
    # Se selecciona por PR-AUC sobre el tramo de CALIBRACION (posterior al watch,
    # anterior al test): nunca se mira 2026 para elegir hiperparametros.
    proba_c = model.predict_proba(X_c)[:, 1]
    pr = average_precision_score(y_c, proba_c)
    resultados.append({**params, "best_iter": model.best_iteration, "pr_auc_calib": pr})
    print(f"  {params} -> best_iter={model.best_iteration}  PR-AUC(calib)={pr:.4f}")

res = pd.DataFrame(resultados).sort_values("pr_auc_calib", ascending=False)
res.to_csv(OUT_DIR / "fase7b_xgb_tuning.csv", index=False)
print("\n=== Top 5 ===")
print(res.head(5).to_string(index=False))
