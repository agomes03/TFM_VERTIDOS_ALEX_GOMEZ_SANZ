"""
Descarga recortes RGB de Sentinel-2 para las mejores parejas y compone una
figura de verificacion visual con el contorno de la mancha detectada por SAR
(Cerulean/Sentinel-1) superpuesto sobre la imagen optica.

Se descartan automaticamente los recortes que caen parcialmente fuera de la
huella de la escena S2 (zonas de nodata en negro), que de otro modo ocupan
media figura sin aportar informacion.
"""
import os
import io
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ee.Initialize(project=os.environ["EE_PROJECT"])

OUT_DIR = Path(__file__).parent / "salida_golfo_persico"
FIG_DIR = Path(__file__).parent / "figuras"
CHIP_DIR = Path(__file__).parent / "chips_opticos"
CHIP_DIR.mkdir(exist_ok=True)

BUFFER_KM = 9
N_PANELES = 6
MAX_NODATA = 0.06     # fraccion maxima de pixeles negros (fuera de huella S2)

INK = "#12201f"
MUTED = "#6b7a7b"
ACCENT = "#eb6834"
WARN = "#c98500"
SOMERO_M = 10  # umbral de aguas someras para marcar el panel

BATIMETRIA = ee.Image("NOAA/NGDC/ETOPO1").select("bedrock")


def fraccion_nodata(im):
    a = np.asarray(im)
    return float((a.sum(axis=2) < 12).mean())


res = pd.read_csv(OUT_DIR / "optico_emparejamiento.csv")
res = res.sort_values(["desfase_horas", "nubes_pct"])

print("Cargando geometrias de las manchas...")
slicks = gpd.read_file(OUT_DIR / "cerulean_golfo_persico_limpio.geojson")
slicks["id"] = slicks["id"].astype(str)

paneles = []
descartados = 0

for r in res.itertuples():
    if len(paneles) >= N_PANELES:
        break
    dlon = BUFFER_KM / (111.0 * np.cos(np.radians(r.lat)))
    dlat = BUFFER_KM / 111.0
    region = [r.lon - dlon, r.lat - dlat, r.lon + dlon, r.lat + dlat]
    aoi = ee.Geometry.Rectangle(region)

    try:
        img = ee.Image(r.s2_image_id).select(["B4", "B3", "B2"])
        url = img.getThumbURL({
            "region": aoi, "dimensions": 640, "format": "png",
            "min": 0, "max": 1800, "gamma": 1.25,
        })
        resp = requests.get(url, timeout=90)
        resp.raise_for_status()
        im = Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"  [warn] {r.slick_id}: {type(e).__name__}")
        continue

    nd = fraccion_nodata(im)
    if nd > MAX_NODATA:
        descartados += 1
        print(f"  [skip] {r.slick_id}: {nd:.0%} fuera de la huella S2")
        continue

    # Batimetria: la profundidad distingue mar abierto de aguas someras, donde
    # el fondo visible y las estructuras sedimentarias favorecen los look-alikes.
    try:
        prof = BATIMETRIA.reduceRegion(
            ee.Reducer.mean(), ee.Geometry.Point([r.lon, r.lat]).buffer(3000), 1000
        ).getInfo()["bedrock"]
    except Exception:
        prof = np.nan

    im.save(CHIP_DIR / f"s2_{r.slick_id}.png")
    geom = slicks.loc[slicks["id"] == str(r.slick_id), "geometry"]
    paneles.append({
        "slick_id": r.slick_id, "img": im, "region": region,
        "geom": geom.iloc[0] if len(geom) else None,
        "desfase": r.desfase_horas, "nubes": r.nubes_pct,
        "area": r.area_km2, "conf": r.machine_confidence,
        "nodata": nd, "profundidad_m": prof,
    })
    print(f"  OK {r.slick_id}  Dt {r.desfase_horas}h  nubes {r.nubes_pct}%  "
          f"area {r.area_km2} km2  profundidad {prof:.0f} m")

print(f"\nPaneles utiles: {len(paneles)} (descartados por nodata: {descartados})")

# ---------- Figura mosaico ----------
# Ordenar: primero mar abierto, al final las aguas someras (el hallazgo)
paneles.sort(key=lambda p: -(p["profundidad_m"] if not np.isnan(p["profundidad_m"]) else -999))

ncol = 3
nrow = int(np.ceil(len(paneles) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 4.75 * nrow))
axes = np.atleast_1d(axes).ravel()

for ax, p in zip(axes, paneles):
    minx, miny, maxx, maxy = p["region"]
    ax.imshow(np.asarray(p["img"]), extent=[minx, maxx, miny, maxy], origin="upper")

    if p["geom"] is not None:
        gpd.GeoSeries([p["geom"]], crs="EPSG:4326").boundary.plot(
            ax=ax, color=ACCENT, linewidth=1.6)

    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.set_xticks([]); ax.set_yticks([])

    prof = p["profundidad_m"]
    somero = (not np.isnan(prof)) and abs(prof) < SOMERO_M
    for s in ax.spines.values():
        s.set_edgecolor(WARN if somero else "#d3dcda")
        s.set_linewidth(2.2 if somero else 1.0)

    # Toda la metainformacion va DENTRO del panel: evita que el texto de una
    # fila se solape con el titulo de la siguiente.
    titulo = f"Δt = {p['desfase']:.1f} h · nubes {p['nubes']:.1f}% · fondo {abs(prof):.0f} m"
    ax.set_title(titulo, fontsize=9.5, fontweight="bold",
                 color=WARN if somero else INK, loc="left", pad=6)
    etiqueta = f"{p['area']:.0f} km²  ·  confianza {p['conf']:.3f}"
    if somero:
        etiqueta += "  ·  AGUAS SOMERAS"
    ax.text(0.025, 0.03, etiqueta,
            transform=ax.transAxes, fontsize=8.5, color="white", va="bottom",
            bbox=dict(boxstyle="round,pad=0.32",
                      facecolor="#8a5a00" if somero else "#12201f",
                      alpha=0.78, edgecolor="none"))

for ax in axes[len(paneles):]:
    ax.axis("off")

fig.suptitle("Verificación óptica: contorno de la mancha detectada en SAR (Sentinel-1) sobre imagen Sentinel-2",
             fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=1.0)
fig.text(0.02, -0.005,
         "Δt = desfase temporal entre la pasada de Sentinel-1 y la de Sentinel-2; «fondo» = profundidad (ETOPO1). El contorno naranja marca la mancha delimitada por Cerulean sobre la imagen\n"
         "radar; la deriva por corriente y viento durante ese intervalo impide una coincidencia exacta, por lo que la comparación es cualitativa. Los dos paneles destacados en ámbar corresponden\n"
         "a aguas de apenas 2 m, donde el fondo visible dificulta la interpretación visual; el análisis del catálogo completo (apartado 6.6) descarta, no obstante, un sesgo sistemático de\n"
         "detección en aguas someras: su tasa por pasada es un 42 % inferior a la de aguas más profundas.",
         fontsize=8.5, color=MUTED, ha="left", va="top")
fig.tight_layout(rect=[0, 0.03, 1, 0.975], h_pad=2.6)
fig.savefig(FIG_DIR / "fig_verificacion_optica.png", dpi=190, bbox_inches="tight")
plt.close(fig)
print("fig_verificacion_optica.png OK")

pd.DataFrame([{k: v for k, v in p.items() if k not in ("img", "geom")} for p in paneles]).to_csv(
    OUT_DIR / "optico_paneles_seleccionados.csv", index=False)
