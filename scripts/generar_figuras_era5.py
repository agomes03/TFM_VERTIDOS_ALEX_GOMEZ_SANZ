"""
Genera todas las figuras (gráficos y mapas) para la memoria del TFM,
a partir de los datos reales ya calculados en las fases anteriores.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, LogNorm

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"
FIG_DIR = Path(__file__).parent / "figuras"
FIG_DIR.mkdir(exist_ok=True)

# ---------- estilo ----------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#c98500"
INK = "#12201f"
MUTED = "#6b7a7b"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

SEQ_COLORS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_COLORS)


def clean_ax(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


# ============================================================
# Figura 1: series mensuales (4 paneles)
# ============================================================
panel = pd.read_csv(OUT_DIR / "panel_modelo_fase6_era5.csv")
monthly = panel.groupby("mes").agg(
    n_pases=("n_pases", "sum"), n_events=("n_events", "sum"),
    wind_speed=("wind_speed", "mean"), ais_hours=("ais_hours", "sum"),
).reset_index()
monthly["tasa"] = monthly["n_events"] / monthly["n_pases"] * 100
months = pd.to_datetime(monthly["mes"])

fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
specs = [
    (axes[0, 0], monthly["n_events"], "Detecciones / mes", BLUE, "{:.0f}"),
    (axes[0, 1], monthly["tasa"], "Tasa de detección / pasada (%)", ORANGE, "{:.1f}"),
    (axes[1, 0], monthly["wind_speed"], "Viento medio (m/s)", AQUA, "{:.1f}"),
    (axes[1, 1], monthly["ais_hours"] / 1000, "Tráfico AIS (miles de horas-buque)", YELLOW, "{:.0f}"),
]
for ax, series, title, color, _ in specs:
    ax.plot(months, series, color=color, linewidth=2)
    ax.fill_between(months, series, series.min(), color=color, alpha=0.08)
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left", color=INK, pad=8)
    ax.tick_params(axis="x", rotation=0)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    clean_ax(ax)

fig.suptitle("Series mensuales, golfo Pérsico (enero 2023 – agosto 2026)", fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=1.01)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(FIG_DIR / "fig_series_mensuales.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_series_mensuales.png OK")

# ============================================================
# Figura 2: GLM rate ratios
# ============================================================
glm = pd.read_csv(OUT_DIR / "glm_fase6_era5_rate_ratios.csv", index_col=0)
glm = glm.drop(index=[i for i in glm.index if i in ("const", "alpha")], errors="ignore")
name_map = {
    "wind_speed": "Viento", "dist_costa_km": "Distancia a costa", "log_ais_hours": "Tráfico AIS (log)",
    "mes_sin": "Estacionalidad (sen)", "mes_cos": "Estacionalidad (cos)", "anios_desde_2023": "Tendencia anual",
}
glm["label"] = [name_map.get(v, v) for v in glm.index]
glm["log_rr"] = np.log(glm["rate_ratio"])
glm = glm.sort_values("log_rr")

fig, ax = plt.subplots(figsize=(7.5, 3.6))
colors = [ORANGE if v < 0 else BLUE for v in glm["log_rr"]]
ax.barh(glm["label"], glm["log_rr"], color=colors, height=0.55)
ax.axvline(0, color=INK, linewidth=1)
for y, (rr, lr) in enumerate(zip(glm["rate_ratio"], glm["log_rr"])):
    ax.text(lr + (0.03 if lr >= 0 else -0.03), y, f"×{rr:.2f}", va="center",
            ha="left" if lr >= 0 else "right", fontsize=9, color=INK)
ax.set_xlabel("log(rate ratio)  ·  la línea vertical marca «sin efecto» (rate ratio = 1)")
ax.set_title("Modelo explicativo (GLM binomial negativa): rate ratios", fontsize=11.5, fontweight="bold", loc="left")
clean_ax(ax)
ax.spines["left"].set_visible(False)
ax.set_xlim(glm["log_rr"].min() - 0.3, glm["log_rr"].max() + 0.35)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_glm_rate_ratios.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_glm_rate_ratios.png OK")

# ============================================================
# Figura 3: RF importancia de variables
# ============================================================
imp = pd.read_csv(OUT_DIR / "fase7_importancia_variables_era5.csv")
name_map_rf = {
    "wind_speed": "Viento", "dist_costa_km": "Distancia a costa", "log_ais_hours": "Tráfico AIS (log)",
    "n_pases": "Nº de pasadas S1", "mes_sin": "Estacionalidad (sen)", "mes_cos": "Estacionalidad (cos)",
    "anios_desde_2023": "Tendencia anual", "lag1_has_event": "Detección mes anterior",
    "lag1_n_events": "Nº detecciones mes anterior",
}
imp["label"] = [name_map_rf.get(v, v) for v in imp["variable"]]
imp = imp.sort_values("importancia")

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.barh(imp["label"], imp["importancia"] * 100, color=AQUA, height=0.55)
for y, v in enumerate(imp["importancia"] * 100):
    ax.text(v + 0.6, y, f"{v:.1f}%", va="center", fontsize=9, color=INK)
ax.set_xlabel("Importancia relativa (%)")
ax.set_title("Modelo predictivo (Random Forest): importancia de variables", fontsize=11.5, fontweight="bold", loc="left")
clean_ax(ax)
ax.spines["left"].set_visible(False)
ax.set_xlim(0, imp["importancia"].max() * 100 * 1.2)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_rf_importancia.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_rf_importancia.png OK")

# ============================================================
# Figura 4: curva precision-recall
# ============================================================
pr = pd.read_csv(OUT_DIR / "fase7_curva_precision_recall_era5.csv")
baseline_rate = 0.0838

fig, ax = plt.subplots(figsize=(5.6, 4.4))
ax.plot(pr["recall"], pr["precision"], color=BLUE, linewidth=2)
ax.axhline(baseline_rate, color=MUTED, linewidth=1, linestyle="--", label=f"Base aleatoria ({baseline_rate:.3f})")
ax.set_xlabel("Recall (sensibilidad)")
ax.set_ylabel("Precisión")
ax.set_title("Curva precisión-recall — validación 2026", fontsize=11.5, fontweight="bold", loc="left")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=9, loc="upper right")
clean_ax(ax)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_pr_curve.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_pr_curve.png OK")

# ============================================================
# Mapas: riesgo previsto e histórico
# ============================================================
grid = gpd.read_file(OUT_DIR / "rejilla_golfo_persico.geojson")
grid["cell_id"] = grid["cell_id"].astype(str)

coast = gpd.read_file(Path(__file__).parent / "ne_10m_coastline.geojson", bbox=(47.5, 23.5, 56.5, 30.5))

pred = pd.read_csv(OUT_DIR / "fase7_predicciones_2026_era5.csv")
pred["cell_id"] = pred["cell_id"].astype(str)
riesgo = pred.groupby("cell_id")["proba_prevista"].mean().reset_index()
grid_riesgo = grid.merge(riesgo, on="cell_id", how="left")

positivos = pd.read_csv(OUT_DIR / "positivos_celda_escena.csv")
positivos["cell_id"] = positivos["cell_id"].astype(str)
hist = positivos.groupby("cell_id")["n_events"].sum().reset_index()
grid_hist = grid.merge(hist, on="cell_id", how="left")


def plot_map(gdf, column, title, cbar_label, log=False, fname=None, p95_cap=False):
    fig, ax = plt.subplots(figsize=(7.5, 5.6))
    vmin = gdf[column].min()
    vmax = gdf[column].quantile(0.95) if p95_cap else gdf[column].max()
    norm = LogNorm(vmin=max(gdf[column][gdf[column] > 0].min(), 1e-6), vmax=vmax) if log else None
    gdf.plot(column=column, cmap=SEQ_CMAP, ax=ax, linewidth=0, missing_kwds={"color": "#f2f5f4"},
              vmin=None if log else vmin, vmax=None if log else vmax, norm=norm)
    coast.plot(ax=ax, color=MUTED, linewidth=0.8)
    ax.set_xlim(47.3, 56.7); ax.set_ylim(23.3, 30.7)
    ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=SEQ_CMAP, norm=norm if log else plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9.5, color=MUTED)
    cbar.ax.tick_params(labelsize=8.5, color=MUTED, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(fname, "OK")


plot_map(grid_riesgo, "proba_prevista", "Mapa de riesgo previsto (2026)", "Probabilidad prevista de detección",
          p95_cap=True, fname="fig_mapa_riesgo.png")
plot_map(grid_hist, "n_events", "Densidad histórica de detecciones (2023–2026)", "Nº de detecciones acumuladas (escala log)",
          log=True, fname="fig_mapa_historico.png")

# ============================================================
# Esquema de la metodología (diagrama de fases)
# ============================================================
fig, ax = plt.subplots(figsize=(9.2, 11.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 21)
ax.axis("off")

box_style = dict(boxstyle="round,pad=0.35,rounding_size=0.12", linewidth=1.3)

phases = [
    ("FASE 0–2", "Descarga y limpieza de detecciones", "Cerulean (SkyTruth) · paginación completa · depuración geométrica", "#dcefee", "#0c6a6c"),
    ("FASE 3", "Rejilla espacio-temporal", "Celdas de 10 km × mes · asignación de detecciones", "#dcefee", "#0c6a6c"),
    ("FASE 1", "Catálogo de esfuerzo", "Escenas Sentinel-1 con y sin detección (Google Earth Engine)", "#e8f3ea", "#1baf7a"),
    ("FASE 4", "Variables explicativas", "Viento (NOAA GFS0P25) · distancia a costa (Natural Earth) · tráfico AIS (Global Fishing Watch)", "#fdefe3", "#eb6834"),
    ("FASE 6", "Modelo explicativo (GLM)", "Poisson / binomial negativa · offset = log(esfuerzo)", "#fdf3d9", "#c98500"),
    ("FASE 7", "Modelo predictivo (Random Forest)", "Validación temporal · calibración isotónica · PR-AUC / ROC-AUC / Brier", "#fdf3d9", "#c98500"),
    ("FASE 8", "Dashboard geoespacial", "Mapas de riesgo e históricos · series temporales · resultados de los modelos", "#e6e9fb", "#4a3aa7"),
]

y = 20.0
box_h = 2.15
gap = 0.55
positions = []
for tag, title, sub, fill, edge in phases:
    y0 = y - box_h
    rect = FancyBboxPatch((0.4, y0), 9.2, box_h, facecolor=fill, edgecolor=edge, **box_style)
    ax.add_patch(rect)
    ax.text(0.85, y0 + box_h - 0.45, tag, fontsize=9.5, fontweight="bold", color=edge, family="monospace")
    ax.text(0.85, y0 + box_h - 0.95, title, fontsize=12.5, fontweight="bold", color=INK)
    ax.text(0.85, y0 + 0.45, sub, fontsize=9.3, color=MUTED, wrap=True)
    positions.append((y0, y0 + box_h))
    y = y0 - gap

for i in range(len(positions) - 1):
    y_top_of_next = positions[i][0]
    y_bottom_of_prev_arrow_start = positions[i][0]
    ax.annotate("", xy=(5, positions[i + 1][1] + 0.02), xytext=(5, positions[i][0] - 0.02),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.4, mutation_scale=14))

fig.tight_layout()
fig.savefig(FIG_DIR / "fig_esquema_metodologia.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_esquema_metodologia.png OK")

print("\nTodas las figuras generadas en:", FIG_DIR)
