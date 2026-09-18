"""
Figuras comparativas Random Forest vs XGBoost:
  - fig_comparativa_metricas.png : metricas lado a lado
  - fig_comparativa_pr.png       : curvas PR superpuestas
  - fig_comparativa_importancia.png : importancia de variables, ambos modelos
  - fig_mapa_riesgo_xgb.png      : mapa de riesgo del mejor modelo
"""
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"
FIG_DIR = Path(__file__).parent / "figuras"
FIG_DIR.mkdir(exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#12201f"
MUTED = "#6b7a7b"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})

SEQ_COLORS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_COLORS)


def clean_ax(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


comp = pd.read_csv(OUT_DIR / "fase7b_comparativa_modelos.csv", index_col=0)
baseline_pr = comp["baseline_pr_auc"].iloc[0]
baseline_brier = comp["baseline_brier"].iloc[0]

# ---------- 1. Metricas lado a lado ----------
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
metricas = [
    ("pr_auc", "PR-AUC", baseline_pr, "mayor es mejor"),
    ("roc_auc", "ROC-AUC", 0.5, "mayor es mejor"),
    ("brier", "Brier score", baseline_brier, "menor es mejor"),
]
modelos = list(comp.index)
colores = [BLUE, ORANGE]

for ax, (col, titulo, base, nota) in zip(axes, metricas):
    vals = comp[col].values
    bars = ax.bar(modelos, vals, color=colores, width=0.55)
    ax.axhline(base, color=MUTED, linestyle="--", linewidth=1)
    ax.text(1.45, base, "baseline", fontsize=8, color=MUTED, va="bottom", ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=INK)
    ax.set_title(f"{titulo}\n", fontsize=11, fontweight="bold", loc="left", color=INK)
    ax.text(0, 1.02, nota, transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.set_ylim(0, max(max(vals), base) * 1.28)
    clean_ax(ax)

fig.suptitle("Random Forest vs XGBoost — validación 2026 (fuera de muestra)",
             fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=1.04)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_comparativa_metricas.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_comparativa_metricas.png OK")

# ---------- 2. Curvas PR superpuestas ----------
pr_rf = pd.read_csv(OUT_DIR / "fase7b_pr_curve_rf.csv")
pr_xgb = pd.read_csv(OUT_DIR / "fase7b_pr_curve_xgb.csv")

fig, ax = plt.subplots(figsize=(5.8, 4.4))
ax.plot(pr_rf["recall"], pr_rf["precision"], color=BLUE, linewidth=2,
        label=f"Random Forest (PR-AUC {comp.loc['Random Forest', 'pr_auc']:.3f})")
ax.plot(pr_xgb["recall"], pr_xgb["precision"], color=ORANGE, linewidth=2,
        label=f"XGBoost (PR-AUC {comp.loc['XGBoost', 'pr_auc']:.3f})")
ax.axhline(baseline_pr, color=MUTED, linewidth=1, linestyle="--",
           label=f"Base aleatoria ({baseline_pr:.3f})")
ax.set_xlabel("Recall (sensibilidad)")
ax.set_ylabel("Precisión")
ax.set_title("Curvas precisión-recall comparadas", fontsize=11.5, fontweight="bold", loc="left")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=9, loc="upper right")
clean_ax(ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_comparativa_pr.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_comparativa_pr.png OK")

# ---------- 3. Importancia de variables, ambos modelos ----------
imp_rf = pd.read_csv(OUT_DIR / "fase7b_importancia_rf.csv")
imp_xgb = pd.read_csv(OUT_DIR / "fase7b_importancia_xgb.csv")

NAMES = {
    "wind_speed": "Viento (ERA5)", "dist_costa_km": "Distancia a costa",
    "log_ais_hours": "Tráfico AIS (log)", "n_pases": "Nº de pasadas S1",
    "mes_sin": "Estacionalidad (sen)", "mes_cos": "Estacionalidad (cos)",
    "anios_desde_2023": "Tendencia anual", "lag1_has_event": "Detección mes anterior",
    "lag1_n_events": "Nº detecciones mes ant.",
}

merged = imp_rf.merge(imp_xgb, on="variable", suffixes=("_rf", "_xgb"))
merged["label"] = merged["variable"].map(NAMES)
merged = merged.sort_values("importancia_rf")

y = np.arange(len(merged))
h = 0.38
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.barh(y + h / 2, merged["importancia_rf"] * 100, height=h, color=BLUE, label="Random Forest")
ax.barh(y - h / 2, merged["importancia_xgb"] * 100, height=h, color=ORANGE, label="XGBoost")
ax.set_yticks(y)
ax.set_yticklabels(merged["label"])
ax.set_xlabel("Importancia relativa (%)")
ax.set_title("Importancia de variables: Random Forest vs XGBoost",
             fontsize=11.5, fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower right")
clean_ax(ax)
ax.spines["left"].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_comparativa_importancia.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_comparativa_importancia.png OK")

# ---------- 4. Mapa de riesgo del mejor modelo ----------
mejor_col = "proba_xgb" if comp.loc["XGBoost", "pr_auc"] >= comp.loc["Random Forest", "pr_auc"] else "proba_rf"
mejor_nombre = "XGBoost" if mejor_col == "proba_xgb" else "Random Forest"
print(f"Mapa de riesgo con: {mejor_nombre}")

pred = pd.read_csv(OUT_DIR / "fase7b_predicciones_2026_ambos.csv")
pred["cell_id"] = pred["cell_id"].astype(str)
riesgo = pred.groupby("cell_id")[mejor_col].mean().reset_index()

grid = gpd.read_file(OUT_DIR / "rejilla_golfo_persico.geojson")
grid["cell_id"] = grid["cell_id"].astype(str)
grid_riesgo = grid.merge(riesgo, on="cell_id", how="left")
coast = gpd.read_file(Path(__file__).parent / "ne_10m_coastline.geojson", bbox=(47.5, 23.5, 56.5, 30.5))

fig, ax = plt.subplots(figsize=(7.5, 5.6))
vmax = grid_riesgo[mejor_col].quantile(0.95)
grid_riesgo.plot(column=mejor_col, cmap=SEQ_CMAP, ax=ax, linewidth=0,
                  vmin=grid_riesgo[mejor_col].min(), vmax=vmax,
                  missing_kwds={"color": "#f2f5f4"})
coast.plot(ax=ax, color=MUTED, linewidth=0.8)
ax.set_xlim(47.3, 56.7); ax.set_ylim(23.3, 30.7)
ax.set_title(f"Mapa de riesgo previsto 2026 — {mejor_nombre}", fontsize=12.5,
             fontweight="bold", loc="left", pad=10)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
sm = plt.cm.ScalarMappable(cmap=SEQ_CMAP,
                            norm=plt.Normalize(vmin=grid_riesgo[mejor_col].min(), vmax=vmax))
cbar = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
cbar.set_label("Probabilidad prevista de detección", fontsize=9.5, color=MUTED)
cbar.ax.tick_params(labelsize=8.5, color=MUTED, labelcolor=MUTED)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_mapa_riesgo_mejor.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_mapa_riesgo_mejor.png OK")

# ---------- 5. Bootstrap de la diferencia de PR-AUC ----------
boot_path = OUT_DIR / "fase7b_bootstrap_diffs.csv"
if boot_path.exists():
    diffs = pd.read_csv(boot_path)["diff"].values
    resumen_boot = pd.read_csv(OUT_DIR / "fase7b_bootstrap_diferencia.csv").iloc[0]
    lo, hi = resumen_boot["ic95_bajo"], resumen_boot["ic95_alto"]
    obs = resumen_boot["diferencia"]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(diffs, bins=45, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=INK, linewidth=1.6, label="Sin diferencia")
    ax.axvline(obs, color=ORANGE, linewidth=2, label=f"Diferencia observada ({obs:+.4f})")
    ax.axvspan(lo, hi, color=MUTED, alpha=0.13, label=f"IC 95% [{lo:+.4f}, {hi:+.4f}]")
    ax.set_xlabel("Diferencia de PR-AUC (XGBoost − Random Forest)")
    ax.set_ylabel("Frecuencia (réplicas bootstrap)")
    ax.set_title("¿Es real la ventaja de XGBoost? — bootstrap pareado sobre el test 2026",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    clean_ax(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_bootstrap_diferencia.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("fig_bootstrap_diferencia.png OK")

print("\nFiguras comparativas generadas en:", FIG_DIR)
