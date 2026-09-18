"""
Esquema de fases de la metodologia para la version final del TFM.

Sustituye a la figura generada en generar_figuras_era5.py: elimina la fase de
dashboard, actualiza el viento a ERA5, recoge la comparacion RF/XGBoost y
cierra el flujo con los analisis de robustez.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG_DIR = Path(__file__).resolve().parent / "figuras"
FIG_DIR.mkdir(exist_ok=True)

INK = "#12201f"
MUTED = "#6b7a7b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "text.color": INK,
})

phases = [
    ("FASE 0–2", "Descarga y limpieza de detecciones",
     "Cerulean (SkyTruth) · paginación completa · depuración geométrica", "#dcefee", "#0c6a6c"),
    ("FASE 3", "Rejilla espacio-temporal",
     "Celdas de 10 km × mes · asignación de detecciones", "#dcefee", "#0c6a6c"),
    ("FASE 1", "Catálogo de esfuerzo",
     "Escenas Sentinel-1 con y sin detección (Google Earth Engine)", "#e8f3ea", "#1baf7a"),
    ("FASE 4", "Variables explicativas",
     "Viento (ERA5, Copernicus) · distancia a costa (Natural Earth) · tráfico AIS (Global Fishing Watch)",
     "#fdefe3", "#eb6834"),
    ("FASE 6", "Modelo explicativo (GLM)",
     "Binomial negativa · offset = log(esfuerzo de observación)", "#fdf3d9", "#c98500"),
    ("FASE 7", "Modelo predictivo",
     "Random Forest y XGBoost · validación temporal · calibración isotónica · bootstrap",
     "#fdf3d9", "#c98500"),
    ("VALIDACIÓN", "Análisis de robustez",
     "Etiquetas humanas · Sentinel-2 · autocorrelación · tamaño de celda · origen de la tendencia",
     "#e6e9fb", "#4a3aa7"),
]

fig, ax = plt.subplots(figsize=(9.2, 11.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 21)
ax.axis("off")

box_style = dict(boxstyle="round,pad=0.35,rounding_size=0.12", linewidth=1.3)

y = 20.0
# el pad del recuadro (0,35 por lado) ocupa 0,7 del hueco: dejarlo holgado
box_h = 1.85
gap = 1.1
positions = []
for tag, title, sub, fill, edge in phases:
    y0 = y - box_h
    ax.add_patch(FancyBboxPatch((0.4, y0), 9.2, box_h, facecolor=fill, edgecolor=edge, **box_style))
    ax.text(0.85, y0 + box_h - 0.45, tag, fontsize=9.5, fontweight="bold", color=edge, family="monospace")
    ax.text(0.85, y0 + box_h - 0.95, title, fontsize=12.5, fontweight="bold", color=INK)
    ax.text(0.85, y0 + 0.45, sub, fontsize=9.3, color=MUTED)
    positions.append((y0, y0 + box_h))
    y = y0 - gap

for i in range(len(positions) - 1):
    ax.annotate("", xy=(5, positions[i + 1][1] + 0.02), xytext=(5, positions[i][0] - 0.02),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.4, mutation_scale=14))

fig.tight_layout()
fig.savefig(FIG_DIR / "fig_esquema_metodologia_v2.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_esquema_metodologia_v2.png OK")
